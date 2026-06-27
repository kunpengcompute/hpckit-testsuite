#include "gtest/gtest.h"
#include <iostream>
#include <fstream>

class CustomTestEventListener : public::testing::EmptyTestEventListener {
public:
    CustomTestEventListener() {
    }

    ~CustomTestEventListener() override {
    }

    void OnTestStart(const::testing::TestInfo& test_info) override {
        std::cout << "start test " <<test_info.name() << std::endl;
    }

    void OnTestEnd(const::testing::TestInfo& test_info) override {
        std::cout << "end test " <<test_info.name() << " PASSED" << std::endl;
    }

    void OnTestProgramEnd(const::testing::UnitTest& unit_test) override {
        int failed = unit_test.failed_test_count();
        int total = unit_test.total_test_count();
        std::cout << total << " tests ran, " << total - failed << " tests PASSED, "
        << failed << " tests FAILED." << std::endl;
    }
};

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);

    auto& listeners = ::testing::UnitTest::GetInstance()->listeners();
    delete listeners.Release(listeners.default_result_printer());
    listeners.Append(new CustomTestEventListener());

    return RUN_ALL_TESTS();
}