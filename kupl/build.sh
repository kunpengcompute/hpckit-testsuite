set -e

export KUPL_PROJ_PATH=$(cd "$(dirname ${0})"; pwd -P)
export KUPL_CLEANUP="on"
export KUPL_COMPILER="gcc"
export ENABLE_KUPL_MMA="OFF"

function build_test()
{
    echo ""
    echo "build test"
    local source_path=$KUPL_PROJ_PATH/test
    local build_path=$KUPL_PROJ_PATH/build
    mkdir -p ${build_path} && cd ${build_path}
    cmake -S ${source_path} -B ${build_path}    \
        -DCMAKE_C_COMPILER=$(which mpicc)       \
        -DCMAKE_CXX_COMPILER=$(which mpicxx)        \
        -DENABLE_KUPL_MMA=${ENABLE_KUPL_MMA}        \
        -DCMAKE_INSTALL_PREFIX=${KUPL_PROJ_PATH}
    cmake --build ${build_path} -j --target install
}

function cleanup()
{
    echo ""
    echo "do cleanup"
    local build_path=$KUPL_PROJ_PATH/build
    if [ -d "${build_path}" ];then
        rm -rf ${build_path}
        echo "cleanup ${build_path}"
    fi
}

function set_compiler()
{
    case "$1" in
        gcc)
            export CC=$(which gcc)
            export CXX=$(which g++)
            KUPL_C_COMPILER=$(which gcc)
            KUPL_CXX_COMPILER=$(which g++)
            ;;
        clang)
            export CC=$(which clang)
            export CXX=$(which clang++)
            KUPL_C_COMPILER=$(which clang)
            KUPL_CXX_COMPILER=$(which clang++)
            ;;
        *)
            echo "Unsupported compiler $1."
            exit 1
            ;;
    esac
    KUPL_COMPILER="$1"
}

function parse_args()
{
    for i in "$@"; do
        case "$i" in
            --cleanup=*)
                if [[ "${i#*=}" == "off" ]]; then
                    KUPL_CLEANUP="off"
                fi
                ;;
            gcc)
                KUPL_COMPILER="gcc"
                ;;
            bisheng)
                KUPL_COMPILER="clang"
                ;;
            *)
                echo "Unknown option: $i"
                echo ""
                exit 1
                ;;
        esac
    done
}

function set_enable_mma()
{
    cputype=$(cat /proc/cpuinfo | grep 'CPU part' | awk 'NR==1{print $4}')
    if [[ "$cputype" == "0xd22" ]]; then
        ENABLE_KUPL_MMA="ON"
    fi
}

function main()
{
    echo "PROJ_PATH:    "   $KUPL_PROJ_PATH
    if [[ "${KUPL_CLEANUP,,}" == "on" ]]; then
        cleanup
    fi
    set_enable_mma
    set_compiler $KUPL_COMPILER
    build_test
}

parse_args $@
main