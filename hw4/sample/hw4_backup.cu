//***********************************************************************************
// Optimized CUDA Bitcoin Miner
// Parallel nonce search using GPU
//***********************************************************************************

#include <iostream>
#include <fstream>
#include <string>

#include <cstdio>
#include <cstring>

#include <cassert>

#include "sha256.h"

////////////////////////   Block   /////////////////////

typedef struct _block
{
    unsigned int version;
    unsigned char prevhash[32];
    unsigned char merkle_root[32];
    unsigned int ntime;
    unsigned int nbits;
    unsigned int nonce;
}HashBlock;


////////////////////////   Utils   ///////////////////////

unsigned char decode(unsigned char c)
{
    switch(c)
    {
        case 'a':
            return 0x0a;
        case 'b':
            return 0x0b;
        case 'c':
            return 0x0c;
        case 'd':
            return 0x0d;
        case 'e':
            return 0x0e;
        case 'f':
            return 0x0f;
        case '0' ... '9':
            return c-'0';
    }
    return 0;
}

void convert_string_to_little_endian_bytes(unsigned char* out, char *in, size_t string_len)
{
    assert(string_len % 2 == 0);

    size_t s = 0;
    size_t b = string_len/2-1;

    for(; s < string_len; s+=2, --b)
    {
        out[b] = (unsigned char)(decode(in[s])<<4) + decode(in[s+1]);
    }
}

void print_hex(unsigned char* hex, size_t len)
{
    for(int i=0;i<len;++i)
    {
        printf("%02x", hex[i]);
    }
}

void print_hex_inverse(unsigned char* hex, size_t len)
{
    for(int i=len-1;i>=0;--i)
    {
        printf("%02x", hex[i]);
    }
}

int little_endian_bit_comparison(const unsigned char *a, const unsigned char *b, size_t byte_len)
{
    for(int i=byte_len-1;i>=0;--i)
    {
        if(a[i] < b[i])
            return -1;
        else if(a[i] > b[i])
            return 1;
    }
    return 0;
}

void getline(char *str, size_t len, FILE *fp)
{
    int i=0;
    while( i<len && (str[i] = fgetc(fp)) != EOF && str[i++] != '\n');
    str[len-1] = '\0';
}

////////////////////////   Hash   ///////////////////////

void double_sha256(SHA256 *sha256_ctx, unsigned char *bytes, size_t len)
{
    SHA256 tmp;
    sha256(&tmp, (BYTE*)bytes, len);
    sha256(sha256_ctx, (BYTE*)&tmp, sizeof(tmp));
}

// Device version of double_sha256
__device__ void double_sha256_device(SHA256 *sha256_ctx, unsigned char *bytes, size_t len)
{
    SHA256 tmp;
    sha256_device(&tmp, (BYTE*)bytes, len);
    sha256_device(sha256_ctx, (BYTE*)&tmp, sizeof(tmp));
}

// Device version of little_endian_bit_comparison
__device__ int little_endian_bit_comparison_device(const unsigned char *a, const unsigned char *b, size_t byte_len)
{
    for(int i=byte_len-1;i>=0;--i)
    {
        if(a[i] < b[i])
            return -1;
        else if(a[i] > b[i])
            return 1;
    }
    return 0;
}

////////////////////   Merkle Root   /////////////////////

void calc_merkle_root(unsigned char *root, int count, char **branch)
{
    size_t total_count = count;
    unsigned char *raw_list = new unsigned char[(total_count+1)*32];
    unsigned char **list = new unsigned char*[total_count+1];

    for(int i=0;i<total_count; ++i)
    {
        list[i] = raw_list + i * 32;
        convert_string_to_little_endian_bytes(list[i], branch[i], 64);
    }

    list[total_count] = raw_list + total_count*32;

    while(total_count > 1)
    {
        int i, j;

        if(total_count % 2 == 1)
        {
            memcpy(list[total_count], list[total_count-1], 32);
        }

        for(i=0, j=0;i<total_count;i+=2, ++j)
        {
            double_sha256((SHA256*)list[j], list[i], 64);
        }

        total_count = j;
    }

    memcpy(root, list[0], 32);

    delete[] raw_list;
    delete[] list;
}

////////////////////////   CUDA Kernel   ///////////////////////

// CUDA Kernel for parallel nonce search
__global__ void find_nonce_kernel(
    HashBlock *blocks,
    unsigned char *target_hex,
    unsigned int *found_nonce,
    int *found_flag,
    unsigned int start_nonce,
    unsigned int total_nonces
)
{
    // Calculate global thread ID
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    
    // Each thread processes multiple nonces
    for(unsigned int offset = tid; offset < total_nonces; offset += stride)
    {
        // Early exit if solution already found
        if(*found_flag)
            return;
        
        unsigned int nonce = start_nonce + offset;
        
        // Create local copy of block
        HashBlock local_block = *blocks;
        local_block.nonce = nonce;
        
        // Calculate double SHA256
        SHA256 sha256_ctx;
        double_sha256_device(&sha256_ctx, (unsigned char*)&local_block, sizeof(HashBlock));
        
        // Check if hash is less than target
        if(little_endian_bit_comparison_device(sha256_ctx.b, target_hex, 32) < 0)
        {
            // Atomic operation to ensure only one thread writes
            int old = atomicCAS(found_flag, 0, 1);
            if(old == 0)
            {
                *found_nonce = nonce;
            }
            return;
        }
    }
}


////////////////////////   Solve Function   ///////////////////////

void solve(FILE *fin, FILE *fout)
{
    // Read data
    char version[9];
    char prevhash[65];
    char ntime[9];
    char nbits[9];
    int tx;
    char *raw_merkle_branch;
    char **merkle_branch;

    getline(version, 9, fin);
    getline(prevhash, 65, fin);
    getline(ntime, 9, fin);
    getline(nbits, 9, fin);
    fscanf(fin, "%d\n", &tx);

    raw_merkle_branch = new char [tx * 65];
    merkle_branch = new char *[tx];
    for(int i=0;i<tx;++i)
    {
        merkle_branch[i] = raw_merkle_branch + i * 65;
        getline(merkle_branch[i], 65, fin);
        merkle_branch[i][64] = '\0';
    }

    // Calculate merkle root
    unsigned char merkle_root[32];
    calc_merkle_root(merkle_root, tx, merkle_branch);

    printf("merkle root(little): ");
    print_hex(merkle_root, 32);
    printf("\n");

    // Prepare block
    HashBlock block;
    convert_string_to_little_endian_bytes((unsigned char *)&block.version, version, 8);
    convert_string_to_little_endian_bytes(block.prevhash, prevhash, 64);
    memcpy(block.merkle_root, merkle_root, 32);
    convert_string_to_little_endian_bytes((unsigned char *)&block.nbits, nbits, 8);
    convert_string_to_little_endian_bytes((unsigned char *)&block.ntime, ntime, 8);
    block.nonce = 0;

    // Calculate target value
    unsigned int exp = block.nbits >> 24;
    unsigned int mant = block.nbits & 0xffffff;
    unsigned char target_hex[32] = {};
    
    unsigned int shift = 8 * (exp - 3);
    unsigned int sb = shift / 8;
    unsigned int rb = shift % 8;
    
    target_hex[sb    ] = (mant << rb);
    target_hex[sb + 1] = (mant >> (8-rb));
    target_hex[sb + 2] = (mant >> (16-rb));
    target_hex[sb + 3] = (mant >> (24-rb));

    printf("Target value (big): ");
    print_hex_inverse(target_hex, 32);
    printf("\n");

    // GPU computation
    HashBlock *d_block;
    unsigned char *d_target;
    unsigned int *d_found_nonce;
    int *d_found_flag;
    
    // Allocate device memory
    cudaMalloc(&d_block, sizeof(HashBlock));
    cudaMalloc(&d_target, 32);
    cudaMalloc(&d_found_nonce, sizeof(unsigned int));
    cudaMalloc(&d_found_flag, sizeof(int));
    
    // Copy data to device
    cudaMemcpy(d_block, &block, sizeof(HashBlock), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, target_hex, 32, cudaMemcpyHostToDevice);
    
    int found_flag = 0;
    unsigned int found_nonce = 0;
    cudaMemcpy(d_found_flag, &found_flag, sizeof(int), cudaMemcpyHostToDevice);
    
    // Launch configuration - optimize for your GPU
    int threadsPerBlock = 256;
    int blocksPerGrid = 1024;  // Adjust based on your GPU
    
    // Search in batches
    unsigned int batch_size = 10000000;  // 10M nonces per batch
    unsigned int start_nonce = 0;
    
    printf("Searching for nonce using GPU...\n");
    
    while(start_nonce <= 0xffffffff && !found_flag)
    {
        unsigned int remaining = 0xffffffff - start_nonce + 1;
        unsigned int current_batch = (remaining < batch_size) ? remaining : batch_size;
        
        // Launch kernel
        find_nonce_kernel<<<blocksPerGrid, threadsPerBlock>>>(
            d_block, d_target, d_found_nonce, d_found_flag,
            start_nonce, current_batch
        );
        
        // Check if solution found
        cudaMemcpy(&found_flag, d_found_flag, sizeof(int), cudaMemcpyDeviceToHost);
        
        if(found_flag)
        {
            cudaMemcpy(&found_nonce, d_found_nonce, sizeof(unsigned int), cudaMemcpyDeviceToHost);
            printf("Found Solution!!\n");
            printf("nonce: %u (0x%08x)\n", found_nonce, found_nonce);
            break;
        }
        
        if(start_nonce % 100000000 == 0 && start_nonce > 0)
        {
            printf("Checked up to nonce #%u\n", start_nonce);
        }
        
        // Check for overflow
        if(start_nonce > 0xffffffff - batch_size)
            break;
            
        start_nonce += batch_size;
    }
    
    // Compute final hash on CPU for verification
    block.nonce = found_nonce;
    SHA256 sha256_ctx;
    double_sha256(&sha256_ctx, (unsigned char*)&block, sizeof(block));
    
    printf("hash(big):    ");
    print_hex_inverse(sha256_ctx.b, 32);
    printf("\n\n");

    // Write result
    for(int i=0;i<4;++i)
    {
        fprintf(fout, "%02x", ((unsigned char*)&block.nonce)[i]);
    }
    fprintf(fout, "\n");

    // Cleanup
    cudaFree(d_block);
    cudaFree(d_target);
    cudaFree(d_found_nonce);
    cudaFree(d_found_flag);
    
    delete[] merkle_branch;
    delete[] raw_merkle_branch;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: hw4 <in> <out>\n");
        return 1;
    }
    
    FILE *fin = fopen(argv[1], "r");
    FILE *fout = fopen(argv[2], "w");

    if(!fin || !fout)
    {
        fprintf(stderr, "Error opening files\n");
        return 1;
    }

    int totalblock;
    fscanf(fin, "%d\n", &totalblock);
    fprintf(fout, "%d\n", totalblock);

    for(int i=0;i<totalblock;++i)
    {
        printf("\n=== Solving block %d/%d ===\n", i+1, totalblock);
        solve(fin, fout);
    }

    fclose(fin);
    fclose(fout);

    return 0;
}
