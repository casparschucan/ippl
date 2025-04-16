#include "Ippl.h"

#include <memory>

#include "PoissonSolvers/FeynmanKacSolver.h"
#include "gtest/gtest.h"

namespace ippl {

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

            mesh       = mesh_type(ownedInput, hx, origin);
            field      = std::make_shared<field_type>(mesh, flayout);
            feynmanKac = ippl::PoissonFeynmanKac<field_type, field_type>(*field, *field);
        }

        flayout_type flayout;
        mesh_type mesh;
        std::shared_ptr<field_type> field;
        PoissonFeynmanKac<field_type, field_type> feynmanKac;
    };

    TEST_F(PoissonFeynmanKacTest, dummy) {
        unsigned expected = 0;
        EXPECT_EQ(feynmanKac.WoS().work, expected);
    }

    TEST_F(PoissonFeynmanKacTest, dummy2) {
        Vector<double, dim> expected = {0.199966, 1.168141, 3.0664};
        Vector<double, dim> result   = feynmanKac.sampleGreenDensity(0.5);
        EXPECT_NEAR(result[0], expected[0], 1e-5);
    }

}  // namespace ippl
// this is required to test the orthotree, as it depends on ippl
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
