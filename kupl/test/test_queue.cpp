#include "gtest/gtest.h"
#include "kupl.h"

static void task_delay(void *args)
{
    sleep(1);
}

TEST(test_queue, kupl_queue_001)
{
    kupl_queue_h q1_ = kupl_queue_create();
    kupl_event_h e_ = kupl_event_create();

    kupl_queue_item_desc_t desc = {
        .field_mask = KUPL_QUEUE_ITEM_DESC_FIELD_NAME,
        .func = task_delay,
        .args = nullptr,
        .name = "delay",
    };
    int ret = kupl_queue_submit(q1_, &desc);
    ASSERT_TRUE(ret == KUPL_OK);
    ret = kupl_event_record(e_, q1_);
    ASSERT_TRUE(ret == KUPL_OK);
    ret = kupl_event_wait(e_);
    ASSERT_TRUE(ret == KUPL_OK);
    kupl_event_destroy(e_);
    kupl_queue_destroy(q1_);
}