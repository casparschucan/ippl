
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
//
//     Example:
//       srun ./TestFeynmanKac 64 64 64 10000 --info 5
//
//

#include "Ippl.h"

#include <Kokkos_MathematicalConstants.hpp>
#include <Kokkos_MathematicalFunctions.hpp>
#include <cstdlib>
#include <cstring>

#include "Utility/IpplException.h"
#include "Utility/IpplTimings.h"

#include "PoissonSolvers/FeynmanKacSolver.h"

KOKKOS_INLINE_FUNCTION double sinRhs(double x, double y, double z) {
    double pi = Kokkos::numbers::pi_v<double>;
    return pi * pi * 3 * Kokkos::sin(pi * x) * Kokkos::sin(pi * y) * Kokkos::sin(pi * z);
}

KOKKOS_INLINE_FUNCTION double sin(double x, double y, double z) {
    double pi = Kokkos::numbers::pi_v<double>;
    return Kokkos::sin(pi * x) * Kokkos::sin(pi * y) * Kokkos::sin(pi * z);
}

int main(int argc, char* argv[]) {
    ippl::initialize(argc, argv);
    {
        Inform msg(argv[0]);
        Inform msg2all(argv[0], INFORM_ALL_NODES);

        const unsigned int Dim = 3;

        using Mesh_t      = ippl::UniformCartesian<double, 3>;
        using Centering_t = Mesh_t::DefaultCentering;
        typedef ippl::Field<double, Dim, Mesh_t, Centering_t> field;
        using Solver_t = ippl::PoissonFeynmanKac<field>;

        // start a timer
        static IpplTimings::TimerRef allTimer = IpplTimings::getTimer("allTimer");
        IpplTimings::startTimer(allTimer);

        // get the gridsize from the user
        ippl::Vector<int, Dim> nr = {std::atoi(argv[1]), std::atoi(argv[2]), std::atoi(argv[3])};

        // get the number of samples from the user
        int N = std::atoi(argv[4]);

        // get the delta
        double delta0 = std::strtod(argv[5], 0);

        // print out info and title for the relative error (L2 norm)
        msg << "Test FeynmanKac, grid = " << nr << " N samples = " << N << " delta0 = " << delta0
            << endl;

        // domain
        ippl::NDIndex<Dim> owned;
        for (unsigned i = 0; i < Dim; i++) {
            owned[i] = ippl::Index(nr[i]);
        }

        // specifies decomposition; here all dimensions are parallel
        std::array<bool, Dim> isParallel;
        isParallel.fill(true);

        // unit box
        double dx                        = 1.0 / nr[0];
        double dy                        = 1.0 / nr[1];
        double dz                        = 1.0 / nr[2];
        ippl::Vector<double, Dim> hr     = {dx, dy, dz};
        ippl::Vector<double, Dim> origin = {0.0, 0.0, 0.0};
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

        // assign the rho field with a gaussian
        auto view_rho    = rho.getView();
        const int nghost = rho.getNghost();
        const auto& ldom = layout.getLocalNDIndex();

        Kokkos::parallel_for(
            "Assign rho field", rho.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                // go from local to global indices
                const int ig = i + ldom[0].first() - nghost;
                const int jg = j + ldom[1].first() - nghost;
                const int kg = k + ldom[2].first() - nghost;

                // define the physical points (cell-centered)
                double x = (ig + 0.5) * hr[0] + origin[0];
                double y = (jg + 0.5) * hr[1] + origin[1];
                double z = (kg + 0.5) * hr[2] + origin[2];

                view_rho(i, j, k) = sinRhs(x, y, z);
            });

        // assign the exact field with its values (erf function)
        auto view_exact = exact.getView();

        Kokkos::parallel_for(
            "Assign exact field", exact.getFieldRangePolicy(),
            KOKKOS_LAMBDA(const int i, const int j, const int k) {
                const int ig = i + ldom[0].first() - nghost;
                const int jg = j + ldom[1].first() - nghost;
                const int kg = k + ldom[2].first() - nghost;

                double x = (ig + 0.5) * hr[0] + origin[0];
                double y = (jg + 0.5) * hr[1] + origin[1];
                double z = (kg + 0.5) * hr[2] + origin[2];

                view_exact(i, j, k) = sin(x, y, z);
            });

        // Parameter List to pass to solver
        ippl::ParameterList params;
        params.add("delta0", delta0);
        params.add("N_samples", N);

        // define an FFTPoissonSolver object
        Solver_t FKsolver(phi, rho, params);  // Solver_t FFTsolver(fieldE, rho, params);

        // iterate over 5 timesteps
        for (int times = 0; times < 5; ++times) {
            // solve the Poisson equation -> rho contains the solution (phi) now
            FKsolver.solve();

            // compute relative error norm for potential
            phi        = phi - exact;
            double err = norm(phi) / norm(exact);

            msg << std::setprecision(16) << norm(phi) << " " << norm(exact) << " " << err << endl;
        }

        // stop the timers
        IpplTimings::stopTimer(allTimer);
        IpplTimings::print();
        IpplTimings::print(std::string("timing.dat"));
    }
    ippl::finalize();

    return 0;
}
