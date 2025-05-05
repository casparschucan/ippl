#include "Ippl.h"

#include <memory>
#include <ostream>

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
            std::array<int, dim> points   = {256, 256, 256};
            std::array<int, dim> lhsPoint = {4, 4, 4};

            ippl::Index Iinput(points[0]);
            ippl::Index Jinput(points[1]);
            ippl::Index Kinput(points[2]);
            ippl::NDIndex<dim> ownedInput(Iinput, Jinput, Kinput);

            ippl::Index IinputLhs(lhsPoint[0]);
            ippl::Index JinputLhs(lhsPoint[1]);
            ippl::Index KinputLhs(lhsPoint[2]);
            ippl::NDIndex<dim> ownedInputLhs(IinputLhs, JinputLhs, KinputLhs);

            // specifies SERIAL, PARALLEL dimensions
            std::array<bool, dim> isParallel;
            isParallel.fill(true);

            flayoutRhs = flayout_type(MPI_COMM_WORLD, ownedInput, isParallel);
            flayoutLhs = flayout_type(MPI_COMM_WORLD, ownedInputLhs, isParallel);

            double dx                  = 1.0 / double(points[0]);
            Vector<double, dim> hx     = {dx, dx, dx};
            Vector<double, dim> origin = {0.0, 0.0, 0.0};

            double dxLhs              = 1.0 / double(lhsPoint[0]);
            Vector<double, dim> hxLhs = {dxLhs, dxLhs, dxLhs};

            meshRhs  = mesh_type(ownedInput, hx, origin);
            fieldRhs = std::make_shared<field_type>(meshRhs, flayoutRhs, 0);
            meshLhs  = mesh_type(ownedInputLhs, hxLhs, origin);
            fieldLhs = std::make_shared<field_type>(meshLhs, flayoutLhs, 0);

            auto field_view  = fieldRhs->getView();
            const int nghost = fieldRhs->getNghost();
            const auto& ldom = flayoutRhs.getLocalNDIndex();
            Kokkos::parallel_for(
                "Assign field", fieldRhs->getFieldRangePolicy(),
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
            feynmanKac = ippl::PoissonFeynmanKac<field_type, field_type>(*fieldLhs, *fieldRhs);
        }

        flayout_type flayoutRhs;
        flayout_type flayoutLhs;
        mesh_type meshRhs;
        mesh_type meshLhs;
        std::shared_ptr<field_type> fieldRhs;
        std::shared_ptr<field_type> fieldLhs;
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

    TEST_F(PoissonFeynmanKacTest, homogeneousWoSPointTest) {
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
                    double result   = feynmanKac.solvePointParallel(x, N);
                    ASSERT_NEAR(result, expected, 1e-1 * expected);
                }
            }
        }
    }

    TEST_F(PoissonFeynmanKacTest, homogeneousWoSTest) {
        feynmanKac.solve();

        auto field_view  = fieldLhs->getView();
        const int nghost = fieldLhs->getNghost();
        const auto& ldom = flayoutLhs.getLocalNDIndex();

        Vector<double, dim> hx(0.25);
        Vector<double, dim> origin(0.0);
        const double pi = Kokkos::numbers::pi_v<double>;

        for (unsigned int i = 0; i < 4; i++) {
            for (unsigned int j = 0; j < 4; j++) {
                for (unsigned int k = 0; k < 4; k++) {
                    double x = (i + 0.5) * hx[0] + origin[0];
                    double y = (j + 0.5) * hx[1] + origin[1];
                    double z = (k + 0.5) * hx[2] + origin[2];

                    double expected = sin(x, y, z) / (pi * pi * 3.0);

                    EXPECT_NEAR(field_view(i, j, k), expected, expected / 10);
                }
            }
        }
    }

    TEST_F(PoissonFeynmanKacTest, samplePointAtLevelTest) {
        size_t N              = 10000;
        Vector<double, dim> x = {0.5, 0.5, 0.5};
        PoissonFeynmanKac<field_type, field_type>::MultilevelSum sample =
            feynmanKac.solvePointAtLevel(x, 1, N);
        std::cout << "sampled with the following results at level 1" << std::endl
                  << "average: " << sample.sampleSum / N << std::endl
                  << "variance: "
                  << (sample.sampleSumSq - sample.sampleSum * sample.sampleSum / N) / N << std::endl
                  << "cost per sample: " << sample.CostSum / N << std::endl;

        sample = feynmanKac.solvePointAtLevel(x, 2, N);
        std::cout << "sampled with the following results at level 2" << std::endl
                  << "average: " << sample.sampleSum / N << std::endl
                  << "variance: "
                  << (sample.sampleSumSq - sample.sampleSum * sample.sampleSum / N) / N << std::endl
                  << "cost per sample: " << sample.CostSum / N << std::endl;
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
