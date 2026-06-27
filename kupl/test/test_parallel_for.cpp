#include <atomic>
#include "gtest/gtest.h"
#include "kupl.h"

static inline void task_int_loop_1D(kupl_nd_range_t *nd_range, void *args, int tid, int tnum)
{
    std::atomic<int> *count = (std::atomic<int> *)args;
    for (int i = nd_range->nd_range[0].lower;
         i < nd_range->nd_range[0].upper; i += nd_range->nd_range[0].step) {
        count->fetch_add(1);
    }
}

TEST(test_pf, kupl_parallel_for_001)
{
    std::atomic<int> count = {0};

    kupl_nd_range_t range;

    int exe8[8] = {0, 1, 2, 3, 4, 5, 6, 7};
    kupl_egroup_h eg8 = kupl_egroup_create(exe8, 8);
    EXPECT_NE(exe8, nullptr);
    kupl_parallel_for_desc_t desc = {
        .field_mask = KUPL_PARALLEL_FOR_DESC_FIELD_DEFAULT,
        .range = &range,
        .egroup = eg8,
        .concurrency = 4,
        .policy = KUPL_LOOP_POLICY_STATIC
    };

    KUPL_1D_RANGE_INIT(range, 1, 6);
    int ret = kupl_parallel_for(&desc, task_int_loop_1D, &count);
    EXPECT_EQ(ret, KUPL_OK);
    EXPECT_EQ(count.load(), 5);
    kupl_egroup_destroy(eg8);
}