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
#include "Kokkos_MathematicalConstants.hpp"
#include "Kokkos_Random.hpp"
#include "Poisson.h"
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
            : random_pool(seed)
            , Base(lhs, rhs) {
            static_assert(std::is_floating_point<Tlhs>::value, "Not a floating point type");
            this->density_max[0] =
                (Dim - 2) * (Dim - 2) / (2 * Dim * (Dim - 1) * Kokkos::pow(Dim - 1, 1 / (Dim - 2)));
            if (Dim == 2) {
                this->density_max = 4 / Kokkos::numbers::e;
            }
            for (unsigned int d = 1; d < Dim; ++d) {
                // calculate the normalization constant which is also the maximum
                // of the angular densities
                // beta function of 1/2 and (n-i)/2 which we express through the
                // gamma function
                const Tlhs Z_i = Kokkos::tgamma(0.5) * Kokkos::tgamma((Dim - d) / 2)
                                 / Kokkos::tgamma((Dim - d + 1) / 2);
                this->density_max[d] = 1 / Z_i;
            }
            setDefaultParameters();
            grid_spacing    = this->rhs_mp->get_mesh().getMeshSpacing();
            grid_sizes      = this->rhs_mp->get_mesh().getGridsize();
            origin          = this->rhs_mp->get_mesh().getOrigin();
            grid_max_bounds = grid_sizes * grid_spacing;
            delta0          = this->params_m.template get<Tlhs>("delta0");
            epsilon         = this->params_m.template get<Tlhs>("tolerance");
        }

        void setSolver(lhs_type lhs) {}

        void solve() override {
            setSolver(*(this->lhs_mp));

            int output = this->params_m.template get<int>("output_type");
            if (output & Base::GRAD) {
                *(this->grad_mp) = -grad(*(this->lhs_mp));
            }
        }

        /**
         * @brief Executes the walk on spheres algorithm from a starting position x0
         * @param x0 starting position
         * @return the integral value and the number of steps taken
         */
        WosSample WoS(Vector_t x0) {
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

                if (distance < delta0) {
                    // if we are close to the boundary, we need to sample the Green's function
                    // density and add it to the sample
                    break;
                }

                // sample the Green's function density
                Vector_t y_j = x + sampleGreenDensity(distance);

                sample.sample += interpolate(y_j);

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
            auto generator = random_pool.get_state();

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

            random_pool.free_state(generator);
            return sphericalToCartesian(sample);
        }

        /**
         * @brief sample the Green's function density for the Poisson equation on a n-Ball in
         * spherical coodrinates
         * @param d radius of the n-Ball we're sampling
         */
        Vector_t sampleGreenDensity(Tlhs d) {
            auto generator = random_pool.get_state();

            Vector_t sample;

            Tlhs y;
            // sample the radius
            Tlhs radius_density_max = density_max[0] * Kokkos::pow(d, 2 * Dim - 1);
            if (Dim == 2) {
                radius_density_max = density_max[0] / d;
            }
            do {
                sample[0] = generator.drand(0, d);
                y         = generator.drand(0, radius_density_max);
            } while (radiusPdf(sample[0], d) < y);

            for (unsigned int i = 1; i < Dim - 1; ++i) {
                // sample the angle using rejection sampling
                do {
                    sample[i] = generator.drand(0, Kokkos::numbers::pi_v<Tlhs>);
                    y         = generator.drand(0, density_max[i]);
                } while (anglePdf(sample[i], i) < y);
            }

            // sample the last angle
            sample[Dim - 1] = generator.drand(0, 2 * Kokkos::numbers::pi_v<Tlhs>);

            random_pool.free_state(generator);
            return sphericalToCartesian(sample);
        }

        /**
         * @brief Function that interpolates the rhs field at the given point
         * @param x point to interpolate
         * @return interpolated value
         */
        KOKKOS_INLINE_FUNCTION Tlhs interpolate(Vector_t x) {
            Tlhs value = 0.0;
            // get index of the nearest gridpoint to the left bottom of x
            Vector_t index = Floor((x - origin) / grid_spacing);
            // get the index of the nearest gridpoint
            for (unsigned int d = 0; d < Dim; ++d) {
                if (x[d] - grid_spacing[d] * index[d] > grid_spacing[d] / 2) {
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
            this->params_m.add("delta0", (Tlhs)1e-3);
        }

        /**
         * @brief Function to sample the Green's function density for the
         * i-th angle for the Poisson equation on a n-Ball
         * @param phi angle
         * @param i index of the angle
         * @return density value
         */
        Tlhs anglePdf(Tlhs phi, unsigned int i) {
            assert(i < Dim - 1 && "invalid function index");
            assert(Dim > 2 && "function only needed for dimension at least 3");
            // calculate the normalization constant
            // beta function of 1/2 and (n-i)/2 which we express through the
            // gamma function
            Tlhs Z_i = Kokkos::tgamma(0.5) * Kokkos::tgamma((Dim - i) / 2)
                       / Kokkos::tgamma((Dim - i + 1) / 2);

            return Kokkos::pow(Kokkos::sin(phi), Dim - i - 1) / Z_i;
        }

        /**
         * @brief Function to sample the Green's function density for the
         * radius for the Poisson equation on a n-Ball
         * @param r radius
         * @param d radius of the n-Ball we're sampling
         * @return density value
         */
        Tlhs radiusPdf(Tlhs r, Tlhs d) {
            if (Dim == 2) {
                return 4 * r / (d * d) * Kokkos::log(d / r);
            }
            assert(r > 0 && r < d && "invalid function index");
            // calculate the normalization constant
            // d^n * (n-2) / 2 * n
            const Tlhs Z_r = Kokkos::pow(d, Dim) * (Dim - 2) / (2 * Dim);

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
        Tlhs getDistanceToBoundary(Vector_t x) {
            Tlhs distance = grid_max_bounds[0];
            for (unsigned int d = 0; d < Dim; ++d) {
                Tlhs dist = Kokkos::min(x[d] - origin[d], grid_max_bounds[d] - x[d]);
                distance  = Kokkos::min(distance, dist);
            }
            return distance;
        }

        /**
         * @brief Function to check if a point is in the domain
         * @param x point to check
         * @return true if the point is in the domain, false otherwise
         */
        bool isInDomain(Vector_t x) {
            for (unsigned int d = 0; d < Dim; ++d) {
                if (x[d] < origin[d] || x[d] > grid_max_bounds[d]) {
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
        Vector_t density_max;

        // the necessary mesh information to compute the walk on sphere samples
        Vector_t grid_spacing;
        Vector_t grid_sizes;
        Vector_t origin;
        Vector_t grid_max_bounds;

        // the Kokkos random number generator pool
        GeneratorPool random_pool;

        Tlhs delta0;
        Tlhs epsilon;
    };
}  // namespace ippl

#endif
