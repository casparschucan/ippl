//
// Class PoissonFeynmanKac
//   Solves the Poisson problem using the Feynman-Kac formula and MLMC
//

#ifndef IPPL_POISSON_FEYNMAN_KAC
#define IPPL_POISSON_FEYNMAN_KAC

#include "Kokkos_Core.hpp"

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

        unsigned int WoS() { return 0; }

    protected:
        void setDefaultParameters() override {
            this->params_m.add("max_levels", 10);
            this->params_m.add("tolerance", (Tlhs)1e-3);
            this->params_m.add("delta0", (Tlhs)1e-3);
        }

        Kokkos::Random_XorShift64_Pool<> random_pool;
    };
}  // namespace ippl

#endif
