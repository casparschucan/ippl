//
// Class PoissonFeynmanKac
//   Solves the Poisson problem using the Feynman-Kac formula and MLMC
//

#ifndef IPPL_POISSON_FEYNMAN_KAC
#define IPPL_POISSON_FEYNMAN_KAC

#include "Kokkos_Core.hpp"

#include <cassert>

#include "Types/Vector.h"

#include "Kokkos_Complex.hpp"
#include "Kokkos_Macros.hpp"
#include "Kokkos_MathematicalConstants.hpp"
#include "Kokkos_Random.hpp"
#include "Poisson.h"
#include "decl/Kokkos_Declare_OPENMP.hpp"
#include "decl/Kokkos_Declare_SERIAL.hpp"

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
        using Vector_t           = ippl::Vector<double, Dim>;

        struct WosSample {
            Tlhs sample;
            WorkType work;
        };

        PoissonFeynmanKac()
            : Base() {
            static_assert(std::is_floating_point<Tlhs>::value, "Not a floating point type");
            setDefaultParameters();
        }

        PoissonFeynmanKac(lhs_type& lhs, rhs_type& rhs, unsigned seed = 42)
            : randomPool_m(seed)
            , Base(lhs, rhs) {
            static_assert(std::is_floating_point<Tlhs>::value, "Not a floating point type");
            this->densityMax_m[0] =
                (Dim - 2) * (Dim - 2) / (2 * Dim * (Dim - 1) * Kokkos::pow(Dim - 1, 1 / (Dim - 2)));
            if (Dim == 2) {
                this->densityMax_m = 4 / Kokkos::numbers::e;
            }
            for (unsigned int d = 1; d < Dim; ++d) {
                // calculate the normalization constant which is also the maximum
                // of the angular densities
                // beta function of 1/2 and (n-i)/2 which we express through the
                // gamma function
                const Tlhs Z_i = Kokkos::tgamma(0.5) * Kokkos::tgamma((Dim - d) / 2.)
                                 / Kokkos::tgamma((Dim - d + 1.) / 2.);
                this->densityMax_m[d] = 1. / Z_i;
            }
            setDefaultParameters();
            gridSpacing_m   = this->rhs_mp->get_mesh().getMeshSpacing();
            gridSizes_m     = this->rhs_mp->get_mesh().getGridsize();
            origin_m        = this->rhs_mp->get_mesh().getOrigin();
            gridMaxBounds_m = gridSizes_m * gridSpacing_m;
            delta0_m        = this->params_m.template get<Tlhs>("delta0");
            epsilon_m       = this->params_m.template get<Tlhs>("tolerance");
        }

        void setSolver(lhs_type lhs) {}

        void solve() override {
            setSolver(*(this->lhs_mp));

            int output = this->params_m.template get<int>("output_type");
            if (output & Base::GRAD) {
                *(this->grad_mp) = -grad(*(this->lhs_mp));
            }
        }

        KOKKOS_INLINE_FUNCTION Tlhs solvePoint(Vector_t x, size_t N) {
            // check if the point is in the domain
            assert(isInDomain(x) && "point is outside the domain");
            // sample the Green's function density
            Tlhs result = 0;
            Kokkos::parallel_reduce(
                "homogeneousWoSTest", N,
                KOKKOS_LAMBDA(const int i, double& val) { val += WoS(x).sample; },
                Kokkos::Sum<double>(result));
            result /= N;
            // interpolate the rhs field at the sampled point
            return result;
        }

        /**
         * @brief Executes the walk on spheres algorithm from a starting position x0
         * @param x0 starting position
         * @return the integral value and the number of steps taken
         */
        KOKKOS_INLINE_FUNCTION WosSample WoS(Vector_t x0) {
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
                sample.sample += sphereVolume_s * distance * distance * interpolate(y_j);

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

            Vector_t sample;

            Tlhs y;
            // set the radius to d
            sample[0] = d;
            // sample the first angle uniformly on the interval [0, pi)
            for (unsigned int i = 1; i < Dim - 1; ++i) {
                // sample the angle using rejection sampling
                sample[i] = generator.drand(0, Kokkos::numbers::pi_v<Tlhs>);
            }
            // sample the last angle uniformly on the interval [0, 2 * pi)
            sample[Dim - 1] = generator.drand(0, 2 * Kokkos::numbers::pi_v<Tlhs>);

            randomPool_m.free_state(generator);
            return sphericalToCartesian(sample);
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
            Tlhs radius_density_max = densityMax_m[0] * Kokkos::pow(d, 2 * Dim - 1);
            if (Dim == 2) {
                radius_density_max = densityMax_m[0] / d;
            }
            do {
                sample[0] = generator.drand(0, d);
                y         = generator.drand(0, radius_density_max);
            } while (radiusPdf(sample[0], d) < y);

            for (unsigned int i = 1; i < Dim - 1; ++i) {
                // sample the angle using rejection sampling
                do {
                    sample[i] = generator.drand(0, Kokkos::numbers::pi_v<Tlhs>);
                    y         = generator.drand(0, densityMax_m[i]);
                } while (anglePdf(sample[i], i) < y);
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
                    std::cout << "index: " << index[d] << " grid_sizes: " << gridSizes_m[d]
                              << std::endl;
                }
                assert(index[d] < gridSizes_m[d] && index[d] >= 0 && "index out of bounds");
            }
            // get the index of the nearest gridpoint
            for (unsigned int d = 0; d < Dim; ++d) {
                if (x[d] - gridSpacing_m[d] * index[d] > gridSpacing_m[d] / 2) {
                    index[d] += 1;
                }
            }
            value = this->rhs_mp->access_with_vector(index, std::make_index_sequence<Dim>{});

            return value;
        }

    protected:
        void setDefaultParameters() override {
            this->params_m.add("max_levels", 10);
            this->params_m.add("tolerance", (Tlhs)1e-3);
            this->params_m.add("delta0", (Tlhs)1e-2);
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
    };
}  // namespace ippl

#endif
