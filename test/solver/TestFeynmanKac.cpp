
// TestFeynmanKac
// This program tests the FeynmanKacSolver class with a homogeneous
// Dirichlet boundary
// The solve is iterated 5 times for the purpose of timing studies.
//   Usage:
//     srun ./TestFeynmanKac <nx> <ny> <nz> <N> <delta0> --info 5
//     nx        = No. cell-centered points in the x-direction
//     ny        = No. cell-centered points in the y-direction
//     nz        = No. cell-centered points in the z-direction
//     N         = No. samples per cell-centered point
//     delta0    = the cutoff distance to the boundary
//
//     Example:
//       srun ./TestFeynmanKac 64 64 64 10000 0.01 --info 5
//
//

#include "Ippl.h"
#include "IpplOperations.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cstdlib>
#include <cstring>
#include <string>

#include "BcTypes.h"

#include "Utility/IpplException.h"
#include "Utility/IpplTimings.h"

#include "ParameterList.h"
#include "PoissonCG.h"
#include "PoissonSolvers/FeynmanKacSolver.h"

template <size_t Dim>
KOKKOS_INLINE_FUNCTION double sinRhs(ippl::Vector<double, Dim> x) {
    double pi  = Kokkos::numbers::pi_v<double>;
    double res = pi * pi * Dim;
    for (unsigned int i = 0; i < Dim; i++) {
        res *= Kokkos::sin(pi * x[i]);
    }
    return res;
}

template <size_t Dim>
KOKKOS_INLINE_FUNCTION double sin(ippl::Vector<double, Dim> x) {
    double pi  = Kokkos::numbers::pi_v<double>;
    double res = 1;
    for (unsigned int i = 0; i < Dim; i++) {
        res *= Kokkos::sin(pi * x[i]);
    }
    return res;
}

template <size_t Dim>
void dimTest(int Nr, int Nsamples, double delta0, Inform& msg) {
    using Mesh_t      = ippl::UniformCartesian<double, Dim>;
    using Centering_t = Mesh_t::DefaultCentering;
    typedef ippl::Field<double, Dim, Mesh_t, Centering_t> field;
    using Solver_t   = ippl::PoissonFeynmanKac<field>;
    using CGSolver_t = ippl::PoissonCG<field>;

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
    Mesh_t mesh(owned, hr, origin);

    // all parallel layout, standard domain, normal axis order
    ippl::FieldLayout<Dim> layout(MPI_COMM_WORLD, owned, isParallel);

    // define the R (rho) field
    field exact, rho;
    exact.initialize(mesh, layout);
    rho.initialize(mesh, layout);

    // define the LHS field
    field phi;
    phi.initialize(mesh, layout);

    typedef ippl::BConds<field, Dim> bc_type;

    bc_type bcField;

    for (unsigned int i = 0; i < 2 * Dim; ++i) {
        bcField[i] = std::make_shared<ippl::ZeroFace<field>>(i);
    }

    phi.setFieldBC(bcField);
    // assign the rho field with a gaussian
    auto view_rho    = rho.getView();
    const int nghost = rho.getNghost();
    const auto& ldom = layout.getLocalNDIndex();

    using index_array_type = typename ippl::RangePolicy<Dim>::index_array_type;
    //// iterate through lhs field and solve at each position
    // ippl::parallel_for(
    //"Assigne lhs based on point evaluation", this->lhs_mp->getFieldRangePolicy(),
    // KOKKOS_LAMBDA(const index_array_type& args) {
    //// local to global index conversion
    // Vector_t xvec =
    //(args + lhsdom.first() - nghost + 0.5) * lhsGridSpacing + lhsorigin;

    //// ippl::apply accesses the view at the given indices and obtains a
    //// reference; see src/Expression/IpplOperations.h
    // ippl::apply(lhsView, args) = solvePoint(xvec, Nsamples_m);
    //});
    Kokkos::parallel_for(
        "Assign rho field", rho.getFieldRangePolicy(), KOKKOS_LAMBDA(const index_array_type& args) {
            // go from local to global indices
            ippl::Vector<double, Dim> xvec = (args + ldom.first() - nghost + 0.5) * dx;

            ippl::apply(view_rho, args) = sinRhs<Dim>(xvec);
        });

    // assign the exact field with its values (erf function)
    auto view_exact = exact.getView();

    Kokkos::parallel_for(
        "Assign exact field", exact.getFieldRangePolicy(),
        KOKKOS_LAMBDA(const index_array_type& args) {
            // go from local to global indices
            ippl::Vector<double, Dim> xvec = (args + ldom.first() - nghost + 0.5) * dx;

            ippl::apply(view_rho, args) = sinRhs<Dim>(xvec);
        });

    // Parameter List to pass to solver
    ippl::ParameterList params;
    params.add("delta0", delta0);
    params.add("N_samples", Nsamples);

    ippl::ParameterList CGparams;

    // define an FFTPoissonSolver object
    Solver_t FKsolver(phi, rho, params);
    CGSolver_t CGSolver(phi, rho);

    std::string Dimstring = std::to_string(Dim);

    static IpplTimings::TimerRef WoSTimer = IpplTimings::getTimer("WosTimer");
    static IpplTimings::TimerRef CGTimer  = IpplTimings::getTimer("CGTimer");

    // iterate over 5 timesteps
    for (int times = 0; times < 5; ++times) {
        IpplTimings::startTimer(WoSTimer);
        // solve the Poisson equation -> rho contains the solution (phi) now
        FKsolver.solve();
        IpplTimings::stopTimer(WoSTimer);
        phi        = phi - exact;
        double err = norm(phi) / norm(exact);

        msg << std::setprecision(16) << norm(phi) << " " << norm(exact) << " " << err << endl;

        IpplTimings::startTimer(CGTimer);

        try {
            CGSolver.solve();
        } catch (IpplException e) {
            msg << e.what() << endl;
            msg << e.where() << endl;
        }
        IpplTimings::stopTimer(CGTimer);

        // compute relative error norm for potential
    }
}

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
        int N = std::atoi(argv[4]);

        // get the delta
        double delta0 = std::strtod(argv[5], 0);

        // print out info and title for the relative error (L2 norm)
        msg << "Test FeynmanKac, grid = " << Nr << " N samples = " << N << " delta0 = " << delta0
            << endl;
        dimTest<1>(Nr, N, delta0, msg);
        // dimTest<2>(Nr, N, delta0, msg);
        // dimTest<3>(Nr, N, delta0, msg);
        // dimTest<4>(Nr, N, delta0, msg);
        //  stop the timers
        IpplTimings::stopTimer(allTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
