output=$(./test_solver < input)
echo "start test kuqcd_solve_multishift"
success=$(echo "$output" | grep -oi 'PASSED' | wc -l)
if [ $success -eq 2 ]; then
    echo "end test kuqcd_solve_multishift PASSED"
    echo "1 tests ran, 1 tests PASSED, 0 tests FAILED"
else
    echo "end test kuqcd_solve_multishift FAILED"
    echo "1 tests ran, 1 tests FAILED, 0 tests PASSED"
fi