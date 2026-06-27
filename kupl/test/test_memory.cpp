#include "gtest/gtest.h"
#include "kupl.h"

void array_1d_init(char *array, size_t sz)
{
    for (size_t i = 0; i < sz; i++) {
        array[i] = (char)i;
    }
}

void kupl_memcpy_test(size_t sz)
{
    char *src = (char *)kupl_malloc(KUPL_MEM_DEFAULT, sizeof(char) * sz);
    char *dst = (char *)kupl_malloc(KUPL_MEM_DEFAULT, sizeof(char) * sz);
    if (src == nullptr || dst == nullptr) {
        return;
    }
    if (sizeof(char) * sz > (1uLL << 32)) {
        int ret = kupl_memcpy(dst, src, sizeof(char) * sz);
        ASSERT_TRUE(ret == KUPL_ERROR);
    } else {
        array_1d_init(src, sz);
        int ret = kupl_memcpy(dst, src, sizeof(char) * sz);
        ASSERT_TRUE(ret == KUPL_OK);
    }
    kupl_free(KUPL_MEM_DEFAULT, src);
    kupl_free(KUPL_MEM_DEFAULT, dst);
}

TEST(test_memory, kupl_memcpy_001)
{
    kupl_memcpy_test(1024);
}