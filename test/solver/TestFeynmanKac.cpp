
// TestFeynmanKac
// This program tests the FeynmanKacSolver class with a homogeneous
// Dirichlet boundary for different dimensions
// The solve is iterated 5 times for the purpose of timing studies.
//   Usage:
//     srun ./TestFeynmanKac <nx> <N> <delta0> --info 5
//     nx        = No. cell-centered points in the each dimension-direction
//     N         = No. samples per cell-centered point
//     delta0    = the cutoff distance to the boundary
//
//     Example:
//       srun ./TestFeynmanKac 64 10000 0.01 --info 5
//
//

#include "Ippl.h"
#include "IpplOperations.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cstdlib>
#include <cstring>
#include <string>
#include <utility>

#include "BcTypes.h"

#include "Utility/IpplException.h"
#include "Utility/IpplTimings.h"

#include "ParameterList.h"
#include "PoissonCG.h"
#include "PoissonSolvers/FeynmanKacSolver.h"
#include "Vector.h"

template <size_t Dim>
class PoissonTesterClass {
public:
    using Mesh_t      = ippl::UniformCartesian<double, Dim>;
    using Centering_t = Mesh_t::DefaultCentering;
    typedef ippl::Field<double, Dim, Mesh_t, Centering_t> field;
    using Solver_t   = ippl::PoissonFeynmanKac<field>;
    using CGSolver_t = ippl::PoissonCG<field>;
    using MLMSample  = typename Solver_t::MultilevelSum;

    field rho_m;
    field exact_m;
    field phi_m;
    Mesh_t mesh_m;
    ippl::FieldLayout<Dim> layout_m;
    Solver_t solver_m;
    CGSolver_t CGSolver_m;
    std::string timerName_m;
    std::string CGtimerName_m;
    double delta0_m;

    // copy constructor
    PoissonTesterClass(const PoissonTesterClass& other)
        : rho_m(other.rho_m)
        , exact_m(other.exact_m)
        , phi_m(other.phi_m)
        , mesh_m(other.mesh_m)
        , layout_m(other.layout_m)
        , solver_m(other.solver_m)
        , timerName_m(other.timerName_m)
        , CGtimerName_m(other.CGtimerName_m)
        , delta0_m(other.delta0_m) {}

    PoissonTesterClass(int Nr, double delta0, int Nsamples)
        : delta0_m(delta0) {
        initialize(Nr, Nsamples);
    }

    void initialize(int Nr, int Nsamples) {
        // get the gridsize from the user
        ippl::Vector<int, Dim> nr(Nr);

        // domain
        ippl::NDIndex<Dim> owned;
        for (unsigned i = 0; i < Dim; i++) {
            owned[i] = ippl::Index(nr[i]);
        }

        // specifies decomposition; here all dimensions are parallel
        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        // unit box
        double dx = 1.0 / nr[0];
        ippl::Vector<double, Dim> hr(dx);
        ippl::Vector<double, Dim> origin(0.0);
        mesh_m = Mesh_t(owned, hr, origin);

        // all parallel layout, standard domain, normal axis order
        layout_m = ippl::FieldLayout<Dim>(MPI_COMM_WORLD, owned, isParallel);
        // define the R (rho) field
        exact_m.initialize(mesh_m, layout_m);
        rho_m.initialize(mesh_m, layout_m);

        // define the LHS field
        phi_m.initialize(mesh_m, layout_m);

        typedef ippl::BConds<field, Dim> bc_type;

        bc_type bcField;

        for (unsigned int i = 0; i < 2 * Dim; ++i) {
            bcField[i] = std::make_shared<ippl::ZeroFace<field>>(i);
        }

        phi_m.setFieldBC(bcField);
        // assign the rho field with a gaussian
        auto view_rho    = rho_m.getView();
        const int nghost = rho_m.getNghost();
        const auto& ldom = layout_m.getLocalNDIndex();

        using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
        ippl::parallel_for(
            "Assign rho field", rho_m.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const index_array_type& args) {
                // go from local to global indices
                ippl::Vector<double, Dim> xvec = (args + ldom.first() - nghost + 0.5) * dx;

                ippl::apply(view_rho, args) = PoissonTesterClass::sinRhs(xvec);
            });

        // assign the exact field with its values (erf function)
        auto view_exact = exact_m.getView();

        ippl::parallel_for(
            "Assign exact field", exact_m.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const index_array_type& args) {
                // go from local to global indices
                ippl::Vector<double, Dim> xvec = (args + ldom.first() - nghost + 0.5) * dx;

                ippl::apply(view_exact, args) = PoissonTesterClass::sin(xvec);
            });
        // Parameter List to pass to solver
        ippl::ParameterList params;
        params.add("delta0", delta0_m);
        params.add("N_samples", Nsamples);

        ippl::ParameterList CGparams;

        // define an FFTPoissonSolver object
        solver_m   = Solver_t(phi_m, rho_m, params);
        CGSolver_m = CGSolver_t(phi_m, rho_m);

        std::string Dimstring = std::to_string(Dim);
        timerName_m           = "WosTimer";
        timerName_m.append(Dimstring);
        CGtimerName_m = "CGTimer";
        CGtimerName_m.append(Dimstring);
    }

    static KOKKOS_INLINE_FUNCTION double sinRhs(ippl::Vector<double, Dim> x) {
        double pi  = Kokkos::numbers::pi_v<double>;
        double res = pi * pi * Dim;
        for (unsigned int i = 0; i < Dim; i++) {
            res *= Kokkos::sin(pi * x[i]);
        }
        return res;
    }

    static KOKKOS_INLINE_FUNCTION double sin(ippl::Vector<double, Dim> x) {
        double pi  = Kokkos::numbers::pi_v<double>;
        double res = 1;
        for (unsigned int i = 0; i < Dim; i++) {
            res *= Kokkos::sin(pi * x[i]);
        }
        return res;
    }
    void dimTest(int Nsamples, Inform& msg) {
        IpplTimings::TimerRef WoSTimer = IpplTimings::getTimer(timerName_m.c_str());
        IpplTimings::TimerRef CGTimer  = IpplTimings::getTimer(CGtimerName_m.c_str());

        ippl::Vector<double, Dim> test_pos(.5);
        // iterate over 5 timesteps
        for (int times = 0; times < 5; ++times) {
            IpplTimings::startTimer(WoSTimer);
            // solve the Poisson equation -> rho contains the solution (phi) now
            solver_m.solvePointParallel(test_pos, Nsamples);
            IpplTimings::stopTimer(WoSTimer);
            phi_m      = phi_m - exact_m;
            double err = norm(phi_m) / norm(exact_m);

            msg << std::setprecision(16) << norm(phi_m) << " " << norm(exact_m) << " " << err
                << endl;

            IpplTimings::startTimer(CGTimer);

            CGSolver_m.solve();
            IpplTimings::stopTimer(CGTimer);

            // compute relative error norm for potential
        }
    }
    void convergenceTest(size_t Nsamples, double delta0, Inform& msg) {
        IpplTimings::TimerRef WoSTimer = IpplTimings::getTimer(timerName_m.c_str());

        ippl::Vector<double, Dim> test_pos(.5);
        // iterate over 5 timesteps
        for (int times = 0; times < 1; ++times) {
            IpplTimings::startTimer(WoSTimer);
            // solve the Poisson equation -> rho contains the solution (phi) now
            double res = solver_m.solvePoint(test_pos, Nsamples);
            IpplTimings::stopTimer(WoSTimer);
            double err = Kokkos::abs(res - sin(test_pos));

            msg << std::setprecision(16) << res << " " << sin(test_pos) << " " << err << endl;
        }
    }

    void MLMCspeedupTest(size_t Nsamples, double epsilon, Inform& msg) {
        IpplTimings::TimerRef WoSTimer = IpplTimings::getTimer(timerName_m.c_str());

        ippl::Vector<double, Dim> test_pos(.5);
        solver_m.updateParameter("tolerance", epsilon);
        // iterate over 5 timesteps
        for (int times = 0; times < 5; ++times) {
            IpplTimings::startTimer(WoSTimer);
            // solve the Poisson equation -> rho contains the solution (phi) now
            solver_m.updateParameter("delta0", delta0_m);
            auto [res, work, maxLevel] = solver_m.solvePointMultilevelWithWork(test_pos);
            IpplTimings::stopTimer(WoSTimer);
            double err = Kokkos::abs(res - sin(test_pos));
            // compute the speedup to normal WoS Poisson
            double deltaTest = epsilon / 2.;
            solver_m.updateParameter("delta0", deltaTest);
            MLMSample pureWoS = solver_m.solvePointAtLevel(test_pos, 0, Nsamples);
            double varL =
                (pureWoS.sampleSumSq - pureWoS.sampleSum * pureWoS.sampleSum / Nsamples) / Nsamples;
            double costL   = pureWoS.CostSum * varL / (epsilon * epsilon * Nsamples);
            double speedup = (double)costL / (double)work;
            msg << std::setprecision(16) << res << " " << sin(test_pos) << " " << err
                << " speedup: " << speedup << " NlCl: " << pureWoS.CostSum << " varL: " << varL
                << " costL " << costL << " costMLMC: " << work << " max Level: " << maxLevel
                << endl;
        }
    }
};

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(argv[0]);
        Inform msg2all(argv[0], INFORM_ALL_NODES);
        // start a timer
        static IpplTimings::TimerRef allTimer = IpplTimings::getTimer("allTimer");
        IpplTimings::startTimer(allTimer);

        int Nr = std::atoi(argv[1]);
        // get the number of samples from the user
        unsigned long N = std::atoi(argv[2]);

        // get the delta
        double delta0  = std::strtod(argv[3], 0);
        double epsilon = std::strtod(argv[4], 0);

        // print out info and title for the relative error (L2 norm)
        msg << "Test FeynmanKac, grid = " << Nr << " N samples = " << N << " delta0 = " << delta0
            << endl;

        PoissonTesterClass<2> twoD(Nr, delta0, N);
        PoissonTesterClass<3> threeD(Nr, delta0, N);
        PoissonTesterClass<4> fourD(Nr, delta0, N);
        // twoD.dimTest(N, msg);
        twoD.MLMCspeedupTest(N, epsilon, msg);
        threeD.MLMCspeedupTest(N, epsilon, msg);
        fourD.MLMCspeedupTest(N, epsilon, msg);
        //   stop the timers
        IpplTimings::stopTimer(allTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
