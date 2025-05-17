//
// Class PoissonFeynmanKac
//   Solves the Poisson problem using the Feynman-Kac formula and MLMC
//

#ifndef IPPL_POISSON_FEYNMAN_KAC
#define IPPL_POISSON_FEYNMAN_KAC

#include "Kokkos_Core.hpp"
#include "Kokkos_Core_fwd.hpp"

#include <cassert>
#include <ostream>
#include <vector>

#include "Types/Vector.h"

#include "Expression/IpplOperations.h"

#include "Kokkos_Complex.hpp"
#include "Kokkos_Macros.hpp"
#include "Kokkos_MathematicalConstants.hpp"
#include "Kokkos_MathematicalFunctions.hpp"
#include "Kokkos_MinMax.hpp"
#include "Kokkos_Random.hpp"
#include "ParameterList.h"
#include "Poisson.h"

namespace ippl {

    template <typename FieldLHS, typename FieldRHS = FieldLHS>
    class PoissonFeynmanKac : public Poisson<FieldLHS, FieldRHS> {
        using Tlhs = typename FieldLHS::value_type;

    public:
        using Base                    = Poisson<FieldLHS, FieldRHS>;
        constexpr static unsigned Dim = FieldLHS::dim;
        using typename Base::lhs_type, typename Base::rhs_type;
        using InverseDiagonalRet = double;
        using DiagRet            = double;
        using WorkType           = unsigned int;
        using GeneratorPool      = typename Kokkos::Random_XorShift64_Pool<>;
        using Vector_t           = ippl::Vector<Tlhs, Dim>;

        struct WosSample {
            Tlhs sample;
            WorkType work;

            KOKKOS_INLINE_FUNCTION WosSample& operator+=(const WosSample& rhs) {
                this->sample += rhs.sample;
                this->work += rhs.work;
                return *this;
            }

            KOKKOS_INLINE_FUNCTION WosSample operator-(const WosSample& rhs) {
                WosSample res;
                res.sample = this->sample - rhs.sample;
                res.work   = Kokkos::max(this->sample, rhs.sample);
                return res;
            }
        };

        struct MultilevelSum {
            Tlhs sampleSum;
            Tlhs sampleSumSq;
            WorkType CostSum;
            size_t Nsamples;

            KOKKOS_INLINE_FUNCTION MultilevelSum& operator+=(const WosSample& sample) {
                this->sampleSum += sample.sample;
                this->sampleSumSq += sample.sample * sample.sample;
                this->CostSum += sample.work;
                this->Nsamples += 1;
                return *this;
            }

            KOKKOS_INLINE_FUNCTION MultilevelSum& operator+=(const MultilevelSum& rhs) {
                this->sampleSum += rhs.sampleSum;
                this->sampleSumSq += rhs.sampleSumSq;
                this->CostSum += rhs.CostSum;
                this->Nsamples += rhs.Nsamples;
                return *this;
            }
        };

        PoissonFeynmanKac()
            : Base() {
            static_assert(std::is_floating_point<Tlhs>::value, "Not a floating point type");
            setDefaultParameters();
        }

        PoissonFeynmanKac(lhs_type& lhs, rhs_type& rhs, unsigned seed = 42)
            : Base(lhs, rhs)
            , randomPool_m(seed) {
            setDefaultParameters();
            initialize();
        }

        PoissonFeynmanKac(lhs_type& lhs, rhs_type& rhs, ParameterList& params, unsigned seed = 42)
            : Base(lhs, rhs)
            , randomPool_m(seed) {
            setDefaultParameters();
            this->params_m.merge(params);
            initialize();
        }

        void initialize() {
            static_assert(std::is_floating_point<Tlhs>::value, "Not a floating point type");
            if (Dim == 2) {
                this->densityMax_m[0] = 4 / Kokkos::numbers::e;
            } else {
                this->densityMax_m[0] =
                    (2 * Dim) / ((Dim - 1) * Kokkos::pow(Dim - 1, 1 / (Dim - 2)));
            }
            for (unsigned int d = 1; d < Dim - 1; ++d) {
                // calculate the normalization constant which is also the maximum
                // of the angular densities
                // beta function of 1/2 and (n-i)/2 which we express through the
                // gamma function
                const Tlhs Z_i = Kokkos::tgamma(0.5) * Kokkos::tgamma((Dim - d) / 2.)
                                 / Kokkos::tgamma((Dim - d + 1.) / 2.);
                this->densityMax_m[d] = 1. / Z_i;
            }
            densityMax_m[Dim - 1] = 1 / (2 * Kokkos::numbers::pi_v<Tlhs>);
            gridSpacing_m         = this->rhs_mp->get_mesh().getMeshSpacing();
            gridSizes_m           = this->rhs_mp->get_mesh().getGridsize();
            origin_m              = this->rhs_mp->get_mesh().getOrigin();
            gridMaxBounds_m       = gridSizes_m * gridSpacing_m;
            delta0_m              = this->params_m.template get<Tlhs>("delta0");
            epsilon_m             = this->params_m.template get<Tlhs>("tolerance");
            Nsamples_m            = this->params_m.template get<int>("N_samples");
        }

        KOKKOS_INLINE_FUNCTION Tlhs sinRhs(Vector_t x) {
            Tlhs pi  = Kokkos::numbers::pi_v<Tlhs>;
            Tlhs res = pi * pi * Dim;
            for (unsigned int i = 0; i < Dim; i++) {
                res *= Kokkos::sin(pi * x[i]);
            }
            return res;
        }

        void solve() override {
            // collect useful parameters from the lhs field
            using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
            auto lhsView           = this->lhs_mp->getView();
            auto lhsdom            = this->lhs_mp->getLayout().getLocalNDIndex();
            auto nghost            = this->lhs_mp->getNghost();
            auto lhsGridSpacing    = this->lhs_mp->get_mesh().getMeshSpacing();
            auto lhsorigin         = this->lhs_mp->get_mesh().getOrigin();

            // iterate through lhs field and solve at each position
            ippl::parallel_for(
                "Assigne lhs based on point evaluation", this->lhs_mp->getFieldRangePolicy(),
                KOKKOS_LAMBDA(const index_array_type& args) {
                    // local to global index conversion
                    Vector_t xvec =
                        (args + lhsdom.first() - nghost + 0.5) * lhsGridSpacing + lhsorigin;

                    // ippl::apply accesses the view at the given indices and obtains a
                    // reference; see src/Expression/IpplOperations.h
                    ippl::apply(lhsView, args) = solvePoint(xvec, Nsamples_m);
                });
            return;
        }

        KOKKOS_INLINE_FUNCTION Tlhs solvePoint(Vector_t x, size_t N) {
            // check if the point is in the domain
            assert(isInDomain(x) && "point is outside the domain");
            // collect N WoS samples and average the results
            Tlhs result = 0;
            for (size_t i = 0; i < N; ++i) {
                result += WoS(x).sample;
            }
            result /= N;
            return result;
        }

        KOKKOS_INLINE_FUNCTION Tlhs solvePointParallel(Vector_t x, size_t N) {
            // check if the point is in the domain
            assert(isInDomain(x) && "point is outside the domain");
            Tlhs partialResult = 0;
            Tlhs result        = 0;
            // for numerical stability
            size_t maxParallelN = 1e5;
            size_t Niter        = Kokkos::max(N / (Tlhs)maxParallelN, (Tlhs)1.);
            // collect N WoS samples and average the results
            for (unsigned i = 0; i < Niter; i++) {
                size_t Nsamples = maxParallelN;
                if (i == Niter - 1) {
                    Nsamples = Kokkos::min(maxParallelN + N % maxParallelN, N);
                }
                // std::cout << "Nsamples: " << Nsamples << std::endl;
                Kokkos::parallel_reduce(
                    "homogeneousWoSTest", Kokkos::RangePolicy<>(0, Nsamples),
                    KOKKOS_LAMBDA(const int /*i*/, Tlhs& val) { val += WoS(x).sample; },
                    Kokkos::Sum<Tlhs>(partialResult));

                result += partialResult / N;
                partialResult = 0;
            }
            return result;
        }

        KOKKOS_INLINE_FUNCTION WosSample correlatedWoS(Vector_t x0, size_t level) {
            assert(level > 0 && "level 0 shouldn't be correlated");
            delta0_m = this->params_m.template get<Tlhs>("delta0");
            WosSample sample;
            sample.work   = 0;
            sample.sample = 0;

            Ncorr++;

            Vector_t x = x0;

            bool coarseIn = true;

            Tlhs delta_ratio = 16;

            Tlhs deltaCoarse = delta0_m / Kokkos::pow(delta_ratio, level - 1);
            Tlhs deltaFine   = deltaCoarse / delta_ratio;

            while (true) {
                Tlhs distance = getDistanceToBoundary(x);
                // sample the offset by sampling the sphere with radius distance
                Vector_t offset = sampleSurface(distance);
                Vector_t x_next = x + offset;
                // check if we are in the domain
                assert(isInDomain(x_next) && "sampled point is outside the domain");

                if (distance < deltaFine) {
                    // if we are close to the boundary, we stop the walk
                    x = x_next;
                    break;
                }
                if (distance < deltaCoarse && coarseIn) {
                    coarseIn = false;
                    Nuncorr++;
                }

                // sample the Green's function density
                Vector_t y_j = x + sampleGreenDensity(distance);

                if (!coarseIn) {
                    sample.sample += sphereVolume_s * distance * distance * sinRhs(y_j);
                }

                // calculate the work done
                sample.work += 2 * Dim;

                x = x_next;
            }
            return sample;
        }

        KOKKOS_FUNCTION MultilevelSum solvePointAtLevel(Vector_t x, size_t level, size_t N) {
            // check if the point is in the domain
            assert(isInDomain(x) && "point is outside the domain");
            // collect N WoS samples and average the results
            MultilevelSum result;
            result.sampleSum   = 0;
            result.sampleSumSq = 0;
            result.CostSum     = 0;
            result.Nsamples    = 0;
            for (size_t i = 0; i < N; ++i) {
                if (level == 0) {
                    result += WoS(x);
                    continue;
                }

                result += correlatedWoS(x, level);
            }
            return result;
        }

        KOKKOS_FUNCTION Tlhs solvePointMultilevel(Vector_t x) {
            size_t maxLevel = this->params_m.template get<int>("max_levels");
            epsilon_m       = this->params_m.template get<Tlhs>("tolerance");
            Kokkos::View<size_t*> Ns("number of samples taken per level", maxLevel);
            Kokkos::View<size_t*> Ndiff("number of samples we need additionally per level",
                                        maxLevel);
            Kokkos::View<Tlhs*> costs("cost per level", maxLevel);
            Kokkos::View<Tlhs*> sum("sum of the samples per level", maxLevel);
            Kokkos::View<Tlhs*> sumSq("sum of the samples per level", maxLevel);

            for (unsigned i = 0; i < maxLevel; ++i) {
                Ns(i)    = 10000;  // this->params_m.template get<size_t>("N_samples");
                Ndiff(i) = Ns(i);
                costs(i) = 0;
                sum(i)   = 0;
                sumSq(i) = 0;
            }

            bool converged     = false;
            size_t curMaxLevel = 3;
            while (!converged && curMaxLevel < maxLevel) {
                // std::cout << "current level: " << curMaxLevel << std::endl;
                //  sample the estimated samples at the current level
                for (unsigned i = 0; i < curMaxLevel; ++i) {
                    MultilevelSum sample = solvePointAtLevel(x, i, Ndiff(i));
                    sum(i) += sample.sampleSum;
                    sumSq(i) += sample.sampleSumSq;
                    costs(i) += sample.CostSum;
                    std::cout << " samples: " << Ns(i) << " average: " << sum(i) / Ns(i)
                              << " sq average: " << sumSq(i) / Ns(i) << std::endl;
                }

                Tlhs varCostSumSq = 0;
                // calculate the sum of the product of variance and cost per sample at each level
                for (unsigned i = 0; i < curMaxLevel; ++i) {
                    Tlhs variance = (sumSq(i) - sum(i) * sum(i) / Ns(i)) / Ns(i);
                    assert(variance > 0 && "variance is negative");
                    if (variance < 0) {
                        // std::cout << "variance too smol: " << variance << "at level: " << i
                        //<< std::endl;
                        variance = 0;
                    }
                    Tlhs costPerSample = costs(i) / Ns(i);

                    varCostSumSq += Kokkos::sqrt(variance * costPerSample);
                }
                std::cout << "get to optimal sample loop";

                // calculate the number of samples we need to take at each level
                for (unsigned i = 0; i < curMaxLevel; ++i) {
                    // calculate the variance
                    Tlhs variance      = (sumSq(i) - sum(i) * sum(i) / Ns(i)) / Ns(i);
                    Tlhs costPerSample = costs(i) / Ns(i);
                    // estimate the number of samples we optimally take
                    size_t optimalNSamples = (Kokkos::sqrt(variance / costPerSample) * varCostSumSq)
                                             / (epsilon_m * epsilon_m);
                    if (optimalNSamples > Ns(i)) {
                        Ndiff(i) = optimalNSamples - Ns(i);
                        Ns(i)    = optimalNSamples;
                    } else {
                        Ndiff(i) = 0;
                    }
                    std::cout << "level: " << i << " additional samples: " << Ndiff(i)
                              << std::flush;
                    // add samples as needed
                    MultilevelSum sample = solvePointAtLevel(x, i, Ndiff(i));
                    // std::cout << sample.sampleSum << std::endl;
                    sum(i) += sample.sampleSum;
                    sumSq(i) += sample.sampleSumSq;
                    costs(i) += sample.CostSum;
                    std::cout << " samples: " << Ns(i) << " average: " << sum(i) / Ns(i)
                              << " sq average: " << sumSq(i) / Ns(i) << std::endl;
                    Ndiff(i) = 0;
                }

                // estimate the convergence rate as the difference between the last two levels
                Tlhs av1 = sum(curMaxLevel - 1) / Ns(curMaxLevel - 1);
                std::vector<Tlhs> logErrs(curMaxLevel - 1);
                for (unsigned i = 0; i < curMaxLevel - 1; ++i) {
                    Tlhs average = sum(i + 1) / Ns(i + 1);
                    logErrs[i]   = Kokkos::log2(average);
                }
                // std::cout << "average: " << av2 << " " << av1 << std::endl;
                Tlhs alpha = -linFit(logErrs);
                // std::cout << "convergence rate: " << alpha << std::endl;
                alpha = Kokkos::max(alpha, (Tlhs)0.5);

                Tlhs estError = Kokkos::abs(av1) / (Kokkos::pow(2, alpha) - 1);
                std::cout << "estimated error: " << estError * 2 << std::endl;
                if (estError * 2 < epsilon_m) {
                    // std::cout << "converged with error: " << estError << std::endl;
                    converged = true;
                } else {
                    // increase the number of levels
                    curMaxLevel++;
                }
            }

            // calculate the final result
            Tlhs result = 0;
            for (unsigned i = 0; i < curMaxLevel; ++i) {
                result += sum(i) / Ns(i);
            }

            // std::cout << "maximal level used: " << curMaxLevel << std::endl;
            return result;
        }

        /**
         * @brief Executes the walk on spheres algorithm from a starting position x0
         * @param x0 starting position
         * @return the integral value and the number of steps taken
         */
        KOKKOS_INLINE_FUNCTION WosSample WoS(Vector_t x0) {
            delta0_m = this->params_m.template get<Tlhs>("delta0");
            WosSample sample;
            sample.work   = 0;
            sample.sample = 0;

            Vector_t x = x0;

            while (true) {
                Tlhs distance = getDistanceToBoundary(x);
                // sample the offset by sampling the sphere with radius distance
                Vector_t offset = sampleSurface(distance);
                Vector_t x_next = x + offset;
                // check if we are in the domain
                assert(isInDomain(x_next) && "sampled point is outside the domain");

                if (distance < delta0_m) {
                    // if we are close to the boundary, we stop the walk
                    x = x_next;
                    break;
                }

                // sample the Green's function density
                Vector_t y_j = x + sampleGreenDensity(distance);
                sample.sample += sphereVolume_s * distance * distance * sinRhs(y_j);

                // calculate the work done
                sample.work += 2 * Dim;

                x = x_next;
            }
            return sample;
        }

        /**
         * @brief Function to sample the surface of the n-Ball with radius d uniformly
         * @param d radius of the n-Ball we're sampling
         * @return a vector of size Dim with the sampled coordinates
         */
        KOKKOS_INLINE_FUNCTION Vector_t sampleSurface(Tlhs d) {
            auto generator = randomPool_m.get_state();

            Vector_t direction;

            // Generate n independent standard normal variables
            for (unsigned i = 0; i < Dim; i++) {
                // generate normal distribution
                direction[i] = generator.normal();
            }

            // Normalize to unit length
            Tlhs norm = Kokkos::sqrt(direction.dot(direction));

            direction *= d / norm;
            randomPool_m.free_state(generator);
            return direction;
        }

        /**
         * @brief sample the Green's function density for the Poisson equation on a n-Ball in
         * spherical coodrinates
         * @param d radius of the n-Ball we're sampling
         */
        KOKKOS_INLINE_FUNCTION Vector_t sampleGreenDensity(Tlhs d) {
            auto generator = randomPool_m.get_state();

            Vector_t sample;

            Tlhs y;
            // sample the radius
            Tlhs radiusDensityMax = densityMax_m[0] / d + (Tlhs)0.1;
            do {
                sample[0] = generator.drand(0, d);
                y         = generator.drand(0, radiusDensityMax);
            } while (radiusPdf(sample[0], d) < y);

            for (unsigned int i = 1; i < Dim - 1; ++i) {
                // sample the angle using rejection sampling
                do {
                    sample[i] = generator.drand(0, Kokkos::numbers::pi_v<Tlhs>);
                    y         = generator.drand(0, densityMax_m[i]);
                } while (anglePdf(sample[i], i) < y);
                // sample[i] = generator.drand(0, Kokkos::numbers::pi_v<Tlhs>);
            }
            // sample the last angle
            sample[Dim - 1] = generator.drand(0, 2 * Kokkos::numbers::pi_v<Tlhs>);

            randomPool_m.free_state(generator);
            return sphericalToCartesian(sample);
        }

        /**
         * @brief Function that interpolates the rhs field at the given point
         * @param x point to interpolate
         * @return interpolated value
         */
        KOKKOS_INLINE_FUNCTION Tlhs interpolate(Vector_t x) {
            Tlhs value = 0.0;

            Vector<size_t, Dim> offset(0.5);
            // get index of the nearest gridpoint to the left bottom of x
            Vector<size_t, Dim> index = Floor((x - origin_m) / gridSpacing_m - offset);
            // check if the index is out of bounds
            for (unsigned int d = 0; d < Dim; ++d) {
                // check if the index is out of bounds
                if (index[d] >= gridSizes_m[d]) {
                    // std::cout << "index: " << index[d] << " grid_sizes: " << gridSizes_m[d]
                    //<< std::endl;
                    index[d]--;
                }
                assert(index[d] < gridSizes_m[d] && index[d] >= 0 && "index out of bounds");
            }
            value = ippl::apply(this->rhs_mp->getView(), index);

            return value;
        }

    protected:
        void setDefaultParameters() override {
            this->params_m.add("max_levels", 10);
            this->params_m.add("N_samples", 1000000000);
            this->params_m.add("tolerance", (Tlhs)1e-3);
            this->params_m.add("delta0", (Tlhs)0.000001);
        }

        /**
         * @brief Function to sample the Green's function density for the
         * i-th angle for the Poisson equation on a n-Ball
         * @param phi angle
         * @param i index of the angle
         * @return density value
         */
        KOKKOS_INLINE_FUNCTION Tlhs anglePdf(Tlhs phi, unsigned int i) {
            assert(i < Dim - 1 && "invalid function index");
            assert(Dim > 2 && "function only needed for dimension at least 3");
            // calculate the normalization constant
            // beta function of 1/2 and (n-i)/2 which we express through the
            // gamma function
            Tlhs Z_i = Kokkos::tgamma(0.5) * Kokkos::tgamma((Dim - i) / 2.)
                       / Kokkos::tgamma((Dim - i + 1.) / 2.);

            return Kokkos::pow(Kokkos::sin(phi), Dim - i - 1.) / Z_i;
        }

        /**
         * @brief Function to sample the Green's function density for the
         * radius for the Poisson equation on a n-Ball
         * @param r radius
         * @param d radius of the n-Ball we're sampling
         * @return density value
         */
        KOKKOS_INLINE_FUNCTION Tlhs radiusPdf(Tlhs r, Tlhs d) {
            if (Dim == 2) {
                return 4. * r / (d * d) * Kokkos::log(d / r);
            }
            assert(r > 0 && r < d && "invalid function index");
            // calculate the normalization constant
            // d^n * (n-2) / 2 * n
            const Tlhs Z_r = Kokkos::pow(d, Dim) * (Dim - 2.) / (2. * Dim);

            // intermediate variables
            const Tlhs r_n2 = Kokkos::pow(r, Dim - 2);
            const Tlhs d_n2 = Kokkos::pow(d, Dim - 2);
            return (d_n2 - r_n2) * r / Z_r;
        }

        /**
         * @brief Function to calculate the distance to the boundary of the domain
         * This assumes that the domain is a hypercube
         * @param x point to check
         * @return distance to the boundary
         */
        KOKKOS_INLINE_FUNCTION Tlhs getDistanceToBoundary(Vector_t x) {
            Tlhs distance = gridMaxBounds_m[0];
            for (unsigned int d = 0; d < Dim; ++d) {
                Tlhs dist = Kokkos::min(x[d] - origin_m[d], gridMaxBounds_m[d] - x[d]);
                distance  = Kokkos::min(distance, dist);
            }
            return distance;
        }

        /**
         * @brief Function to check if a point is in the domain
         * @param x point to check
         * @return true if the point is in the domain, false otherwise
         */
        KOKKOS_INLINE_FUNCTION bool isInDomain(Vector_t x) {
            for (unsigned int d = 0; d < Dim; ++d) {
                if (x[d] < origin_m[d] || x[d] > gridMaxBounds_m[d]) {
                    return false;
                }
            }
            return true;
        }

        /**
         * @brief Function to convert a point in spherical coordinates to Cartesian coordinates
         * @param spherical point in spherical coordinates
         * @return point in Cartesian coordinates
         */
        KOKKOS_INLINE_FUNCTION Vector_t sphericalToCartesian(Vector_t spherical) {
            Vector_t cartesian;
            for (unsigned int d = 0; d < Dim - 1; ++d) {
                cartesian[d] = spherical[0] * Kokkos::cos(spherical[d + 1]);
                for (unsigned int i = 1; i <= d; ++i) {
                    cartesian[d] *= Kokkos::sin(spherical[i]);
                }
            }
            cartesian[Dim - 1] = spherical[0];
            for (unsigned int i = 1; i < Dim; ++i) {
                cartesian[Dim - 1] *= Kokkos::sin(spherical[i]);
            }
            return cartesian;
        }

        /**
         * @brief Function to fit a line to the data
         * @param logErrs vector of log2 errors
         * @return slope of the line
         */
        KOKKOS_FUNCTION Tlhs linFit(std::vector<Tlhs> logErrs) {
            Tlhs sumX  = 0;
            Tlhs sumY  = 0;
            Tlhs sumXY = 0;
            Tlhs sumX2 = 0;
            for (unsigned i = 0; i < logErrs.size(); ++i) {
                sumX += i;
                sumY += logErrs.at(i);
                sumXY += i * logErrs.at(i);
                sumX2 += i * i;
            }
            Tlhs slope =
                (logErrs.size() * sumXY - sumX * sumY) / (logErrs.size() * sumX2 - sumX * sumX);
            return slope;
        }

        // The maxima of the different edge densities for the coordinates
        Vector_t densityMax_m;

        // the necessary mesh information to compute the walk on sphere samples
        Vector_t gridSpacing_m;
        Vector_t gridSizes_m;
        Vector_t origin_m;
        Vector_t gridMaxBounds_m;

        // the Kokkos random number generator pool
        GeneratorPool randomPool_m;

        Tlhs delta0_m;
        Tlhs epsilon_m;
        // the integral of green's function over the unit sphere
        constexpr static Tlhs sphereVolume_s = 1.0 / (2.0 * Dim);

        // Number of samples per point
        size_t Nsamples_m;

        size_t Ncorr   = 0;
        size_t Nuncorr = 0;
    };
}  // namespace ippl

#endif
