#include "gtest/gtest.h"
#include "kupl.h"

kupl_sgraph_h sgraph;
kupl_sgraph_node_h task_gnode_1;
kupl_sgraph_node_h task_gnode_2;
kupl_sgraph_node_h task_gnode_3;

static inline void task_str(void *args)
{
}

static kupl_sgraph_node_h add_node(kupl_sgraph_h sgraph, const char *text)
{
    kupl_sgraph_node_desc_t node_desc = {
        .func = task_str,
        .args = (void *)text,
    };
    return kupl_sgraph_add_node(sgraph, &node_desc);
}

TEST(test_sgraph, kupl_sgraph_001)
{
    sgraph = kupl_sgraph_create();
    ASSERT_TRUE(sgraph != nullptr);

    task_gnode_1 = add_node(sgraph, nullptr);
    ASSERT_TRUE(task_gnode_1 != nullptr);
    task_gnode_2 = add_node(sgraph, nullptr);
    ASSERT_TRUE(task_gnode_2 != nullptr);
    task_gnode_3 = add_node(sgraph, nullptr);
    ASSERT_TRUE(task_gnode_3 != nullptr);

    int ret = kupl_sgraph_add_dep(task_gnode_1, task_gnode_2);
    ASSERT_TRUE(ret == KUPL_OK);
    ret = kupl_sgraph_add_dep(task_gnode_1, task_gnode_3);
    ASSERT_TRUE(ret == KUPL_OK);

    auto graph = kupl_graph_create(nullptr);
    kupl_sgraph_task_desc_t desc = {
        .sgraph = sgraph
    };
    kupl_task_info_t info = {
        .type = KUPL_TASK_TYPE_SGRAPH,
        .desc = &desc,
    };
    ret = kupl_graph_submit(graph, &info);
    ASSERT_TRUE(ret == KUPL_OK);
    kupl_graph_wait(graph);
    kupl_graph_destroy(graph);
    kupl_sgraph_destroy(sgraph);
}