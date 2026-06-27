set -e

export KUTACC_PROJ_PATH=$(cd "$(dirname ${0})"; pwd -P)
export KUTACC_CLEANUP="on"
export KUTACC_COMPILER="gcc"

function build_test()
{
    echo ""
    echo "build test"
    local source_path=$KUTACC_PROJ_PATH/test
    local build_path=$KUTACC_PROJ_PATH/build
    mkdir -p ${build_path} && cd ${build_path}
    cmake -S ${source_path} -B ${build_path}         \
        -DCMAKE_C_COMPILER=${KUTACC_C_COMPILER}      \
        -DCMAKE_CXX_COMPILER=${KUTACC_CXX_COMPILER}  \
        -DCMAKE_INSTALL_PREFIX=${KUTACC_PROJ_PATH}
    cmake --build ${build_path} -j --target install
}

function cleanup()
{
    echo ""
    echo "do cleanup"
    local build_path=$KUTACC_PROJ_PATH/build
    if [ -d "${build_path}" ];then
        rm -rf ${build_path}
        echo "cleanup ${build_path}"
    fi
}

function set_compiler()
{
    case "$1" in
        bisheng)
            export CC=$(which clang)
            export CXX=$(which clang++)
            KUTACC_C_COMPILER=$(which clang)
            KUTACC_CXX_COMPILER=$(which clang++)
            ;;
        *)
            echo "Unsupported compiler $1."
            exit 1
            ;;
    esac
    KUTACC_COMPILER="$1"
}

function parse_args()
{
    for i in "$@"; do
        case "$i" in
            bisheng)
                KUTACC_COMPILER="$i"
                ;;
            gcc)
                KUTACC_COMPILER="$i"
                ;;
            *)
                echo "Unknown option: $i"
                echo ""
                exit 1
                ;;
        esac
    done
}

function main()
{
    echo "PROJ_PATH:    "   $KUTACC_PROJ_PATH
    if [[ "${KUTACC_CLEANUP,,}" == "on" ]]; then
        cleanup
    fi
    set_compiler $KUTACC_COMPILER
    build_test
}

cputype=$(cat /proc/cpuinfo | grep 'CPU part' | awk 'NR==1{print $4}')
if [[ "$cputype" == "0xd22" ]]; then
    parse_args $@
    if [[ "${KUTACC_COMPILER}" == "bisheng" ]]; then
        main
    fi
fi
