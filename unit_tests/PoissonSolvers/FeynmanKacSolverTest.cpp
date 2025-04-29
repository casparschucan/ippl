#include "Ippl.h"

#include <memory>

#include "Kokkos_Macros.hpp"
#include "Kokkos_MathematicalFunctions.hpp"
#include "PoissonSolvers/FeynmanKacSolver.h"
#include "decl/Kokkos_Declare_SERIAL.hpp"
#include "gtest/gtest.h"

namespace ippl {

    KOKKOS_INLINE_FUNCTION double sin(double x, double y, double z) {
        double pi = Kokkos::numbers::pi_v<double>;
        return pi * pi * 3 * Kokkos::sin(pi * x) * Kokkos::sin(pi * y) * Kokkos::sin(pi * z);
    }

    class PoissonFeynmanKacTest : public ::testing::Test {
    public:
        constexpr static unsigned int dim = 3;
        using value_type                  = double;

        using mesh_type      = ippl::UniformCartesian<double, 3>;
        using centering_type = typename mesh_type::DefaultCentering;
        using field_type     = ippl::Field<double, 3, mesh_type, centering_type>;
        using flayout_type   = ippl::FieldLayout<3>;
        PoissonFeynmanKacTest() {
            std::array<int, dim> points = {256, 256, 256};

            ippl::Index Iinput(points[0]);
            ippl::Index Jinput(points[1]);
            ippl::Index Kinput(points[2]);
            ippl::NDIndex<dim> ownedInput(Iinput, Jinput, Kinput);

            // specifies SERIAL, PARALLEL dimensions
            std::array<bool, dim> isParallel;
            isParallel.fill(true);

            flayout = flayout_type(MPI_COMM_WORLD, ownedInput, isParallel);

            double dx                  = 1.0 / double(points[0]);
            Vector<double, dim> hx     = {dx, dx, dx};
            Vector<double, dim> origin = {0.0, 0.0, 0.0};

            mesh             = mesh_type(ownedInput, hx, origin);
            field            = std::make_shared<field_type>(mesh, flayout, 0);
            auto field_view  = field->getView();
            const int nghost = field->getNghost();
            const auto& ldom = flayout.getLocalNDIndex();
            Kokkos::parallel_for(
                "Assign field", field->getFieldRangePolicy(),
                KOKKOS_LAMBDA(const int i, const int j, const int k) {
                    // go from local to global indices
                    const int ig = i + ldom[0].first() - nghost;
                    const int jg = j + ldom[1].first() - nghost;
                    const int kg = k + ldom[2].first() - nghost;

                    // define the physical points (cell-centered)
                    double x = (ig + 0.5) * hx[0] + origin[0];
                    double y = (jg + 0.5) * hx[1] + origin[1];
                    double z = (kg + 0.5) * hx[2] + origin[2];

                    field_view(i, j, k) = sin(x, y, z);
                });
            feynmanKac = ippl::PoissonFeynmanKac<field_type, field_type>(*field, *field);
        }

        flayout_type flayout;
        mesh_type mesh;
        std::shared_ptr<field_type> field;
        PoissonFeynmanKac<field_type, field_type> feynmanKac;
    };

    TEST_F(PoissonFeynmanKacTest, seededWoS) {
        unsigned expected                                           = 90;
        Vector<double, dim> x                                       = {0.5, 0.5, 0.5};
        PoissonFeynmanKac<field_type, field_type>::WosSample sample = feynmanKac.WoS(x);
        EXPECT_EQ(sample.work, expected);
        EXPECT_NEAR(sample.sample, 1.6335841, 1e-5);
    }

    TEST_F(PoissonFeynmanKacTest, density_seeded) {
        Vector<double, dim> expected_sample = {0.199966, 1.0701, 3.9722};
        Vector<double, dim> expected_result;
        expected_result[0] = expected_sample[0] * Kokkos::cos(expected_sample[1]);
        expected_result[1] =
            expected_sample[0] * Kokkos::sin(expected_sample[1]) * Kokkos::cos(expected_sample[2]);
        expected_result[2] =
            expected_sample[0] * Kokkos::sin(expected_sample[1]) * Kokkos::sin(expected_sample[2]);
        Vector<double, dim> result = feynmanKac.sampleGreenDensity(0.5);
        EXPECT_NEAR(result[0], result[0], 1e-5);
    }

    TEST_F(PoissonFeynmanKacTest, homogeneousWoSTest) {
        Vector<double, dim> x;
        const double pi = Kokkos::numbers::pi_v<double>;
        for (unsigned i = 1; i < 5; i++) {
            x[0] = i / 5.0;
            for (unsigned j = 1; j < 5; j++) {
                x[1] = i / 5.0;
                for (unsigned k = 1; k < 5; k++) {
                    x[2]            = i / 5.0;
                    double expected = sin(x[0], x[1], x[2]) / (pi * pi * 3);
                    size_t N        = 1e5;
                    double result   = feynmanKac.solvePoint(x, N);
                    ASSERT_NEAR(result, expected, 1e-1 * expected);
                }
            }
        }
    }

    TEST_F(PoissonFeynmanKacTest, homogeneousWoSTestcorner) {
        Vector<double, dim> x = {0.25, 0.25, 0.25};
        double expected       = std::pow(Kokkos::sin(Kokkos::numbers::pi_v<double> / 4), 3);
        double result         = 0;
        size_t N              = 1e5;
        result                = feynmanKac.solvePoint(x, N);
        EXPECT_NEAR(result, expected, 1e-1);
    }

}  // namespace ippl
// this is necessary to initialize ippl and the (unused but required) MPI
int main(int argc, char** argv) {
    // Initialize MPI and IPPL
    ippl::initialize(argc, argv, MPI_COMM_WORLD);

    // Initialize Google Test
    ::testing::InitGoogleTest(&argc, argv);

    // Run all tests
    int result = RUN_ALL_TESTS();

    // Finalize IPPL and MPI
    ippl::finalize();

    return result;
}
