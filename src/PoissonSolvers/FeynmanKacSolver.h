//
// Class PoissonFeynmanKac
//   Solves the Poisson problem using the Feynman-Kac formula and MLMC
//

#ifndef IPPL_POISSON_FEYNMAN_KAC
#define IPPL_POISSON_FEYNMAN_KAC

#include "Kokkos_Core.hpp"

#include <cassert>

#include "Types/Vector.h"

#include "Kokkos_Random.hpp"
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
        }

        void setSolver(lhs_type lhs) {}

        void solve() override {
            setSolver(*(this->lhs_mp));

            int output = this->params_m.template get<int>("output_type");
            if (output & Base::GRAD) {
                *(this->grad_mp) = -grad(*(this->lhs_mp));
            }
        }

        WosSample WoS() { return {0, 0}; }

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
            do {
                sample[0] = generator.drand(0, d);
                y         = generator.drand(0, density_max[0] * Kokkos::pow(d, 2 * Dim - 1));
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
            return sample;
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
            assert(r > 0 && r < d && "invalid function index");
            // calculate the normalization constant
            // d^n * (n-2) / 2 * n
            const Tlhs Z_r = Kokkos::pow(d, Dim) * (Dim - 2) / (2 * Dim);

            // intermediate variables
            const Tlhs r_n2 = Kokkos::pow(r, Dim - 2);
            const Tlhs d_n2 = Kokkos::pow(d, Dim - 2);
            return (d_n2 - r_n2) * r / Z_r;
        }

        Vector_t density_max;

        GeneratorPool random_pool;
    };
}  // namespace ippl

#endif
