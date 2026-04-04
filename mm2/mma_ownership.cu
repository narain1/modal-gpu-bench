#include <cuda_runtime.h>
#include <cstdio>

// Print the MMA ownership map for m16n8k16
__global__ void mma_ownership_map() {
    int tid = threadIdx.x;
    int warp_id = tid / 32;
    int lane_id = tid % 32;

    // Threads are split into groups of 4 for MMA
    int group_id = lane_id / 4;      // 0-7
    int group_lane = lane_id % 4;    // 0-3

    // Each group handles 2 rows of the 16-row M tile
    // mWarp determines which 16-row section this warp handles
    int m_warp = 16 * warp_id;       // warp 0 -> rows 0-15, warp 1 -> rows 16-31

    // nWarp determines which 8-column section
    int n_warp = 8 * warp_id;        // for our single warp, 0

    // Rows this thread group computes
    int row_0 = m_warp + group_id;          // first output row
    int row_1 = m_warp + group_id + 8;     // second output row (+8 offset)

    // Columns this thread computes (2 consecutive columns per thread)
    int col_0 = n_warp + 2 * group_lane;  // first output column
    int col_1 = n_warp + 2 * group_lane + 1; // second output column

    printf("Thread %2d | Lane %2d | Group %d | grpLane %d | "
           "Rows: [%2d,%2d] | Cols: [%2d,%2d]\n",
           tid, lane_id, group_id, group_lane,
           row_0, row_1, col_0, col_1);
}

__global__ void show_group_details() {
    int lane_id = threadIdx.x % 32;
    int group_id = lane_id / 4;
    int group_lane = lane_id % 4;

    printf("Group %d (threads %d-%d): group_lane=%d -> loads columns %d,%d\n",
           group_id, group_id*4, group_id*4+3, group_lane,
           group_lane*2, group_lane*2+1);
}

int main() {
    printf("=== MMA Ownership Map (m16n8k16) ===\n\n");

    printf("Each warp has 32 threads split into 8 groups of 4 threads.\n");
    printf("Each group computes 2 output rows and 2 output columns.\n\n");

    printf("--- Full Warp Ownership ---\n");
    mma_ownership_map<<<1, 32>>>();
    cudaDeviceSynchronize();

    printf("\n--- Group Details ---\n");
    show_group_details<<<1, 32>>>();
    cudaDeviceSynchronize();

    printf("\n=== Summary ===\n");
    printf("Total output: 16 rows x 8 columns = 128 elements per warp\n");
    printf("Per group (4 threads): 4 rows x 2 cols = 8 elements\n");
    printf("Per thread: 2 rows x 2 cols = 4 elements\n");
    printf("  -> Each thread produces 4 output values (2x2 block)\n");

    return 0;
}
