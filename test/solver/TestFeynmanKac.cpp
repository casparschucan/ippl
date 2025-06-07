
// TestFeynmanKac
// This program tests the FeynmanKacSolver class with a homogeneous
// Dirichlet boundary for different dimensions
// The solve is iterated 5 times for the purpose of timing studies.
//   Usage:
//     srun ./TestFeynmanKac <nx> <N> <delta0> <epsilon> <deltaRatio> <testType> --info 5
//     nx          = No. cell-centered points in the each dimension-direction
//     N           = No. samples per cell-centered point
//     delta0      = the cutoff distance to the boundary
//     epsilon     = the tolerance for the convergence of the multilevel method
//     deltaRatio  = the ratio of the cutoff distance between two levels
//     testType    = the type of test to run options are:
//                   CGComparison, convergenceTest, mlmcSpeedup
//     random      = whether the test position is selected at random.
//                   assumed to be false if none fiven
//
//     Example:
//       srun ./TestFeynmanKac 64 10000 0.01 1e-3 16 mlmcSpeedup --info 5
//
//

#include "Kokkos_Core.hpp"
#include "Ippl.h"
#include "IpplOperations.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <random>
#include <string>
#include <utility>

#include "BcTypes.h"

#include "Utility/IpplException.h"
#include "Utility/IpplTimings.h"

#include "Kokkos_Macros.hpp"
#include "Kokkos_Random.hpp"
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
    std::string mlmctimerName_m;
    double delta0_m;
    double deltaRatio_m;

    ippl::Vector<double, Dim> testPosition_m;

    // copy constructor
    PoissonTesterClass(const PoissonTesterClass& other)
        : rho_m(other.rho_m)
        , exact_m(other.exact_m)
        , phi_m(other.phi_m)
        , mesh_m(other.mesh_m)
        , layout_m(other.layout_m)
        , solver_m(other.solver_m)
        , CGSolver_m(other.phi_m, other.rho_m)
        , timerName_m(other.timerName_m)
        , CGtimerName_m(other.CGtimerName_m)
        , mlmctimerName_m(other.mlmctimerName_m)
        , delta0_m(other.delta0_m)
        , deltaRatio_m(other.deltaRatio_m)
        , testPosition_m(other.testPosition_m) {}

    PoissonTesterClass(int Nr, double delta0, double deltaRatio, int Nsamples)
        : delta0_m(delta0)
        , deltaRatio_m(deltaRatio)
        , testPosition_m(0.5) {
        initialize(Nr, Nsamples);
    }

    PoissonTesterClass(int Nr, double delta0, double deltaRatio, int Nsamples,
                       ippl::Vector<double, Dim> testPosition)
        : delta0_m(delta0)
        , deltaRatio_m(deltaRatio)
        , testPosition_m(testPosition) {
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
        mlmctimerName_m = "MLMCTimer";
        mlmctimerName_m.append(Dimstring);
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

    static KOKKOS_INLINE_FUNCTION double gaussian(ippl::Vector<double, Dim> x) {
        double r2 = 0;
        for (unsigned i = 0; i < Dim; i++) {
            r2 += (x[i] - 0.5) * (x[i] - 0.5);
        }
        r2 *= 100;
        return Kokkos::exp(-r2);
    }

    static KOKKOS_INLINE_FUNCTION double gaussianRhs(ippl::Vector<double, Dim> x) {
        double r2 = 0;
        for (unsigned i = 0; i < Dim; i++) {
            r2 += (x[i] - 0.5) * (x[i] - 0.5);
        }
        r2 *= 100;
        return -2.0 * (r2 - 1.0) * Kokkos::exp(-r2) * 100;
    }

    void CGComparison(double epsilon, Inform& msg) {
        IpplTimings::TimerRef MLMCTimer = IpplTimings::getTimer(mlmctimerName_m.c_str());
        IpplTimings::TimerRef CGTimer   = IpplTimings::getTimer(CGtimerName_m.c_str());

        solver_m.updateParameter("tolerance", epsilon);

        msg << std::setw(20) << "mlmcErr," << std::setw(20) << "mlmcTime," << std::setw(20)
            << "CGErrPoint," << std::setw(20) << "CGErrRelL2," << std::setw(20) << "CGTime" << endl;
        // iterate over 5 timesteps
        for (int times = 0; times < 5; ++times) {
            // reset the MLMC timer
            IpplTimings::infoTimer(mlmctimerName_m.c_str())->wallTime = 0;
            // time the MLMC solve at the test position
            IpplTimings::startTimer(MLMCTimer);
            // solve the Poisson equation -> rho contains the solution (phi) now
            double result = solver_m.solvePointMultilevel(testPosition_m);
            IpplTimings::stopTimer(MLMCTimer);

            // calculate the error
            double MLMCtime = IpplTimings::infoTimer(mlmctimerName_m.c_str())->wallTime;
            double err      = Kokkos::abs(result - sin(testPosition_m));

            // reset CG timer
            IpplTimings::infoTimer(CGtimerName_m.c_str())->wallTime = 0;
            // time theCG solve
            IpplTimings::startTimer(CGTimer);

            CGSolver_m.solve();
            IpplTimings::stopTimer(CGTimer);
            double CGtime = IpplTimings::infoTimer(CGtimerName_m.c_str())->wallTime;

            // calculate the error at the test position
            ippl::Vector<size_t, Dim> index =
                ippl::Floor((testPosition_m - mesh_m.getOrigin()) / mesh_m.getMeshSpacing() - 0.5);

            auto phiViewMirror = Kokkos::create_mirror_view(phi_m.getView());
            Kokkos::deep_copy(phiViewMirror, phi_m.getView());
            double CGres = ippl::apply(phiViewMirror, index);
            double CGerr = Kokkos::abs(CGres - sin(testPosition_m));

            // calculate the relative L2 error
            phi_m        = phi_m - exact_m;
            double L2err = norm(phi_m) / norm(exact_m);

            msg << std::setw(20) << err << "," << std::setw(20) << MLMCtime << "," << std::setw(20)
                << CGerr << "," << std::setw(20) << L2err << "," << std::setw(20) << CGtime << endl;
            // compute relative error norm for potential
        }
    }
    void convergenceTest(size_t Nsamples, Inform& msg) {
        IpplTimings::TimerRef WoSTimer = IpplTimings::getTimer(timerName_m.c_str());

        // iterate over 5 timesteps
        for (int times = 0; times < 1; ++times) {
            IpplTimings::startTimer(WoSTimer);
            // solve the Poisson equation -> rho contains the solution (phi) now
            double res = solver_m.solvePoint(testPosition_m, Nsamples);
            IpplTimings::stopTimer(WoSTimer);
            double err = Kokkos::abs(res - sin(testPosition_m));

            msg << std::setprecision(16) << res << " " << sin(testPosition_m) << " " << err << endl;
        }
    }

    void MLMCspeedupTest(size_t Nsamples, double epsilon, Inform& msg) {
        IpplTimings::TimerRef WoSTimer  = IpplTimings::getTimer(timerName_m.c_str());
        IpplTimings::TimerRef MLMCTimer = IpplTimings::getTimer(mlmctimerName_m.c_str());
        solver_m.updateParameter("tolerance", epsilon);
        solver_m.updateParameter("deltaRatio", deltaRatio_m);
        msg << std::setw(20) << "epsilon," << std::setw(20) << "Dimension," << std::setw(20)
            << "MLMC result," << std::setw(20) << "MC result," << std::setw(20) << "expected,"
            << std::setw(20) << "MLMC cost," << std::setw(20) << "MC cost," << std::setw(20)
            << "max Level," << std::setw(20) << "varL," << endl;
        // iterate over 5 timesteps
        for (int times = 0; times < 5; ++times) {
            IpplTimings::startTimer(MLMCTimer);
            // solve the Poisson equation -> rho contains the solution (phi) now
            solver_m.updateParameter("delta0", delta0_m);
            auto [res, work, maxLevel] = solver_m.solvePointMultilevelWithWork(testPosition_m);
            IpplTimings::stopTimer(MLMCTimer);
            // compute the speedup to normal WoS Poisson
            double deltaTest = delta0_m / (Kokkos::pow(deltaRatio_m, maxLevel));
            solver_m.updateParameter("delta0", deltaTest);
            MLMSample pureWoS = solver_m.solvePointAtLevel(testPosition_m, 0, Nsamples);
            double varL =
                (pureWoS.sampleSumSq - pureWoS.sampleSum * pureWoS.sampleSum / Nsamples) / Nsamples;

            // estimate the cost for standard MC at epsilon precision
            double costL = pureWoS.CostSum * varL / (epsilon * epsilon * Nsamples);

            // solve to tolerance without mlmc
            IpplTimings::startTimer(WoSTimer);
            double MCresult = solver_m.solvePointToTolerance(testPosition_m);
            IpplTimings::stopTimer(WoSTimer);

            // print table based comparison
            msg << std::setprecision(16) << std::setw(20) << epsilon << "," << std::setw(20) << Dim
                << "," << std::setw(20) << res << "," << std::setw(20) << MCresult << ","
                << std::setw(20) << sin(testPosition_m) << "," << std::setw(20) << work << ","
                << std::setw(20) << std::ceil(costL) << "," << std::setw(20) << maxLevel << ","
                << std::setw(20) << varL << "," << endl;
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
        double delta0     = std::strtod(argv[3], 0);
        double epsilon    = std::strtod(argv[4], 0);
        double deltaRatio = std::strtod(argv[5], 0);
        std::string testType(argv[6]);

        bool random = false;

        if (argc >= 8 && std::atoi(argv[7])) {
            random = true;
        }

        ippl::Vector<double, 2> testPos2(0.5);
        ippl::Vector<double, 3> testPos3(0.5);
        ippl::Vector<double, 4> testPos4(0.5);
        ippl::Vector<double, 5> testPos5(0.5);

        if (random) {
            std::random_device rd;
            std::mt19937_64 generator(rd());
            std::uniform_real_distribution<> dist(0.0, 1.0);
            for (unsigned i = 0; i < 5; i++) {
                double rand = dist(generator);
                if (i < 2)
                    testPos2[i] = rand;
                if (i < 3)
                    testPos3[i] = rand;
                if (i < 4)
                    testPos4[i] = rand;

                testPos5[i] = rand;
            }
        }

        // print out info and title for the relative error (L2 norm)
        msg << "Test FeynmanKac, grid = " << Nr << " N samples = " << N << " delta0 = " << delta0
            << " epsilon = " << epsilon << " test type = " << testType
            << " testPosition = " << testPos5 << endl;

        if (testType == "CGComparison") {
            msg << "CG Comparison test" << endl;
            PoissonTesterClass<2> twoD(Nr, delta0, deltaRatio, N, testPos2);
            PoissonTesterClass<3> threeD(Nr, delta0, deltaRatio, N, testPos3);
            PoissonTesterClass<4> fourD(Nr, delta0, deltaRatio, N, testPos4);
            twoD.CGComparison(epsilon, msg);
            threeD.CGComparison(epsilon, msg);
            fourD.CGComparison(epsilon, msg);
        } else if (testType == "convergenceTest") {
            msg << "WoS Convergence test" << endl;
            PoissonTesterClass<2> twoD(Nr, delta0, deltaRatio, N, testPos2);
            PoissonTesterClass<3> threeD(Nr, delta0, deltaRatio, N, testPos3);
            PoissonTesterClass<4> fourD(Nr, delta0, deltaRatio, N, testPos4);
            PoissonTesterClass<5> fiveD(Nr, delta0, deltaRatio, N, testPos5);
            twoD.convergenceTest(N, msg);
            threeD.convergenceTest(N, msg);
            fourD.convergenceTest(N, msg);
            fiveD.convergenceTest(N, msg);
        } else if (testType == "mlmcSpeedup") {
            msg << "MLMC speedup test" << endl;
            PoissonTesterClass<2> twoD(Nr, delta0, deltaRatio, N, testPos2);
            PoissonTesterClass<3> threeD(Nr, delta0, deltaRatio, N, testPos3);
            PoissonTesterClass<4> fourD(Nr, delta0, deltaRatio, N, testPos4);
            PoissonTesterClass<5> fiveD(Nr, delta0, deltaRatio, N, testPos5);
            twoD.MLMCspeedupTest(N, epsilon, msg);
            threeD.MLMCspeedupTest(N, epsilon, msg);
            fourD.MLMCspeedupTest(N, epsilon, msg);
            fiveD.MLMCspeedupTest(N, epsilon, msg);
        } else {
            std::cout << "Unknown test type: " << testType << std::endl;
        }
        //   stop the timers
        IpplTimings::stopTimer(allTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
