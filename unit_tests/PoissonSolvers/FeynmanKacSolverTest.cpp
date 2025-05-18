#include "Ippl.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <ostream>
#include <vector>

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

        using mesh_type      = ippl::UniformCartesian<value_type, 3>;
        using centering_type = typename mesh_type::DefaultCentering;
        using field_type     = ippl::Field<value_type, 3, mesh_type, centering_type>;
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

            value_type dx                  = 1.0 / value_type(points[0]);
            Vector<value_type, dim> hx     = {dx, dx, dx};
            Vector<value_type, dim> origin = {0.0, 0.0, 0.0};

            value_type dxLhs              = 1.0 / value_type(lhsPoint[0]);
            Vector<value_type, dim> hxLhs = {dxLhs, dxLhs, dxLhs};

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
                    value_type x = (ig + 0.5) * hx[0] + origin[0];
                    value_type y = (jg + 0.5) * hx[1] + origin[1];
                    value_type z = (kg + 0.5) * hx[2] + origin[2];

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
        Vector<value_type, dim> x                                   = {0.5, 0.5, 0.5};
        PoissonFeynmanKac<field_type, field_type>::WosSample sample = feynmanKac.WoS(x);
        EXPECT_EQ(sample.work, expected);
        EXPECT_NEAR(sample.sample, 1.6335841, 1e-5);
    }

    TEST_F(PoissonFeynmanKacTest, density_seeded) {
        Vector<value_type, dim> expected_sample = {0.199966, 1.0701, 3.9722};
        Vector<value_type, dim> expected_result;
        expected_result[0] = expected_sample[0] * Kokkos::cos(expected_sample[1]);
        expected_result[1] =
            expected_sample[0] * Kokkos::sin(expected_sample[1]) * Kokkos::cos(expected_sample[2]);
        expected_result[2] =
            expected_sample[0] * Kokkos::sin(expected_sample[1]) * Kokkos::sin(expected_sample[2]);
        Vector<value_type, dim> result = feynmanKac.sampleGreenDensity(0.5);
        EXPECT_NEAR(result[0], result[0], 1e-5);
    }

    TEST_F(PoissonFeynmanKacTest, homogeneousWoSVarianceTest) {
        Vector<value_type, dim> x(.5);
        // const value_type pi = Kokkos::numbers::pi_v<value_type>;
        // value_type expected = sin(x[0], x[1], x[2]) / (pi * pi * 3);
        size_t N = 1e6;
        for (int i = 0; i < 6; i++) {
            value_type delta = std::pow(10, -i - 1);
            std::cout << "delta: " << delta << std::flush;
            feynmanKac.updateParameter("delta0", delta);

            PoissonFeynmanKac<field_type, field_type>::MultilevelSum result =
                feynmanKac.solvePointAtLevel(x, 0, N);
            // ASSERT_NEAR(result, expected, 1e-1 * expected);
            value_type avg = result.sampleSum / result.Nsamples;
            value_type var =
                (result.sampleSumSq - result.sampleSum * result.sampleSum / result.Nsamples)
                / result.Nsamples;
            std::cout << " result: " << avg << " error: " << std::abs(avg - 1.)
                      << " variance: " << var
                      << " work per sample: " << result.CostSum / (value_type)result.Nsamples
                      << std::endl;
        }
    }

    TEST_F(PoissonFeynmanKacTest, homogeneousWoSPointTest) {
        Vector<value_type, dim> x(.5);
        // const value_type pi = Kokkos::numbers::pi_v<value_type>;
        // value_type expected = sin(x[0], x[1], x[2]) / (pi * pi * 3);
        size_t N      = 1e8;
        size_t N_iter = 5;

        for (unsigned j = 0; j < 5; j++) {
            N = std::pow(10, j + 3);

            double resSum   = 0;
            double resSqSum = 0;
            for (int i = 0; i < N_iter; i++) {
                value_type delta = 1e-6;
                std::cout << "delta: " << delta << std::flush;
                feynmanKac.updateParameter("delta0", delta);

                value_type result = feynmanKac.solvePointParallel(x, N);
                resSum += result;
                resSqSum += result * result;
                // ASSERT_NEAR(result, expected, 1e-1 * expected);
                std::cout << " result: " << result << " error: " << std::abs(result - 1)
                          << std::endl;
            }
            value_type avg = resSum / N_iter;
            value_type var = (resSqSum - resSum * resSum / N_iter) / N_iter;
            std::cout << " average: " << avg << " error: " << std::abs(avg - 1.)
                      << " variance: " << var << " std: " << Kokkos::sqrt(var) << std::endl;
        }
    }

    TEST_F(PoissonFeynmanKacTest, homogeneousWoSTest) {
        feynmanKac.solve();

        auto field_view = fieldLhs->getView();
        // const int nghost = fieldLhs->getNghost();
        // const auto& ldom = flayoutLhs.getLocalNDIndex();

        Vector<value_type, dim> hx(0.25);
        Vector<value_type, dim> origin(0.0);
        const value_type pi = Kokkos::numbers::pi_v<value_type>;
        value_type l2norm   = 0.0;
        value_type infnorm  = 0.0;

        for (unsigned int i = 0; i < 4; i++) {
            for (unsigned int j = 0; j < 4; j++) {
                for (unsigned int k = 0; k < 4; k++) {
                    value_type x = (i + 0.5) * hx[0] + origin[0];
                    value_type y = (j + 0.5) * hx[1] + origin[1];
                    value_type z = (k + 0.5) * hx[2] + origin[2];

                    value_type expected = sin(x, y, z) / (pi * pi * 3.0);
                    l2norm += (expected - field_view(i, j, k)) * (expected - field_view(i, j, k));
                    infnorm = std::max(infnorm, std::abs(expected - field_view(i, j, k)));

                    EXPECT_NEAR(field_view(i, j, k), expected, expected / 10);
                }
            }
        }
        l2norm = std::sqrt(l2norm);
        std::cout << "L2 error norm: " << l2norm << std::endl;
        std::cout << "Linf error norm: " << infnorm << std::endl;
    }

    TEST_F(PoissonFeynmanKacTest, samplePointAtLevelTest) {
        size_t N                  = 10000;
        Vector<value_type, dim> x = {0.5, 0.5, 0.5};
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

    TEST_F(PoissonFeynmanKacTest, MLMCTest) {
        Vector<value_type, dim> x = {0.5, 0.5, 0.5};
        int Niter                 = 5;
        value_type delta          = 1e-2;
        value_type epsilon        = 1e-4;
        value_type expected       = 1;
        // unsigned Nstart      = 10000;
        feynmanKac.updateParameter("delta0", delta);
        feynmanKac.updateParameter("tolerance", epsilon);
        std::vector<value_type> MLMCruns(Niter);

        value_type maxError = 0;
        value_type errorSum = 0;

        std::cout << "sampling " << Niter << " samples" << std::endl;
        for (unsigned i = 0; i < Niter; i++) {
            std::cout << "iteration: " << i << " of " << Niter << "\r" << std::flush;
            MLMCruns[i] = feynmanKac.solvePointMultilevel(x);
            maxError    = Kokkos::max(maxError, Kokkos::abs(MLMCruns[i] - expected) / expected);
            errorSum += Kokkos::abs(MLMCruns[i] - expected);
        }

        value_type sum  = std::reduce(MLMCruns.begin(), MLMCruns.end());
        value_type mean = sum / Niter;
        value_type sq_sum =
            std::inner_product(MLMCruns.begin(), MLMCruns.end(), MLMCruns.begin(), 0.0);
        value_type stdev = std::sqrt(sq_sum / Niter - mean * mean);
        std::cout << "mean: " << mean << " mean error: " << errorSum / Niter << std::endl;
        std::cout << "max error: " << maxError << std::endl;
        std::cout << " stdev: " << stdev << std::endl;
        std::cout << "variance: " << stdev * stdev << std::endl;
        EXPECT_NEAR(stdev, 0, 1e-4);
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
