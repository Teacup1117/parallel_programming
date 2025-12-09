#include <iostream>
#include <fstream>
#include <string>
#include <filesystem>
#include <chrono>
#include <omp.h>
#include <mpi.h>

#include "image.hpp"
#include "sift.hpp"


int main(int argc, char *argv[])
{
    // 初始化 MPI
    MPI_Init(&argc, &argv);
    
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    // 優化 OpenMP 設置以獲得最佳負載平衡
    // 禁用動態調整，確保使用固定線程數
    omp_set_dynamic(0);
    
    // 設置嵌套並行層級（已替代 omp_set_nested）
    omp_set_max_active_levels(2);
    
    // 每個 MPI rank 使用所有分配的核心
    int num_threads = omp_get_max_threads();
    omp_set_num_threads(num_threads);
    
    if (argc != 4) {
        if (rank == 0) {
            std::cerr << "Usage: ./hw2 ./testcases/xx.jpg ./results/xx.jpg ./results/xx.txt\n";
        }
        MPI_Finalize();
        return 1;
    }

    std::string input_img = argv[1];
    std::string output_img = argv[2];
    std::string output_txt = argv[3];
    
    auto start = std::chrono::high_resolution_clock::now();
    
    // 讀取並廣播圖像到所有進程
    Image img;
    if (rank == 0) {
        img = Image(input_img);
        img = img.channels == 1 ? img : rgb_to_grayscale(img);
    }
    
    // 廣播圖像尺寸
    int img_info[3] = {0, 0, 0};
    if (rank == 0) {
        img_info[0] = img.width;
        img_info[1] = img.height;
        img_info[2] = img.size;
    }
    MPI_Bcast(img_info, 3, MPI_INT, 0, MPI_COMM_WORLD);
    
    // 其他進程創建圖像容器
    if (rank != 0) {
        img = Image(img_info[0], img_info[1], 1);
    }
    
    // 廣播圖像數據
    MPI_Bcast(img.data, img_info[2], MPI_FLOAT, 0, MPI_COMM_WORLD);
    
    // 所有進程並行計算
    std::vector<Keypoint> kps = find_keypoints_and_descriptors_mpi(img, rank, size);
    
    // 只有 rank 0 負責輸出
    if (rank == 0) {
        /////////////////////////////////////////////////////////////
        // The following code is for the validation
        // You can not change the logic of the following code, because it is used for judge system
        std::ofstream ofs(output_txt);
        if (!ofs) {
            std::cerr << "Failed to open " << output_txt << " for writing.\n";
        } else {
            ofs << kps.size() << "\n";
            for (const auto& kp : kps) {
                ofs << kp.i << " " << kp.j << " " << kp.octave << " " << kp.scale << " ";
                for (size_t i = 0; i < kp.descriptor.size(); ++i) {
                    ofs << " " << static_cast<int>(kp.descriptor[i]);
                }
                ofs << "\n";
            }
            ofs.close();
        }

        Image result = draw_keypoints(img, kps);
        result.save(output_img);
        /////////////////////////////////////////////////////////////

        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> duration = end - start;
        std::cout << "Execution time: " << duration.count() << " ms\n";
        
        std::cout << "Found " << kps.size() << " keypoints.\n";
    }
    
    // 同步所有進程
    MPI_Barrier(MPI_COMM_WORLD);
    
    // 結束 MPI
    MPI_Finalize();
    return 0;
}