#ifdef USE_ASCEND_BACKEND

#include "../neuralnet/nninterface.h"

#include "acl/acl.h"
#include "acl/acl_rt.h"
#include "aclnn/aclnn_base.h"
// ACLNN operator headers (from aclnnop directory)
// Note: inplace variants are declared in the same header as non-inplace
#include "aclnn_convolution.h"
#include "aclnn_relu.h"
#include "aclnn_add.h"
#include "aclnn_mul.h"
#include "aclnn_matmul.h"
// Removed: aclnn_adaptive_avg_pool2d.h - using aclnnMean instead
// Removed: aclnnop/aclnn_adaptive_max_pool2d.h - using aclnnAmax instead
#include "aclnnop/aclnn_amax.h"
#include "aclnnop/aclnn_mean.h"
#include "aclnn_cast.h"
#include "aclnn_fill_scalar.h"
#include "aclnn_copy.h"
#include "aclnn_cat.h"
#include "aclnn_batch_norm.h"
#include "aclnn_mish.h"
#include "aclnnop/aclnn_mul.h"
#include "aclnn_softplus.h"
#include "aclnn_tanh.h"

#include "../neuralnet/desc.h"
#include "../neuralnet/modelversion.h"
#include "../neuralnet/nninputs.h"
#include "../neuralnet/nneval.h"
#include "../neuralnet/activations.h"

#include "../core/test.h"

using namespace std;

// ACLNN_SUCCESS is 0 on CANN 8.3
#ifndef ACLNN_SUCCESS
#define ACLNN_SUCCESS 0
#endif

//---------------------------------------------------------------------------------
// Ascend NPU Backend for KataGo
//
// This backend provides inference acceleration on Huawei Ascend 910 Pro A NPUs
// using the Ascend Computing Language (AscendCL) and ACLNN operators.
//
// Hardware Notes:
// - No bf16 support - use fp16 or fp32 only
// - FP16 is the performance sweet spot - DaVinci Cube Unit optimized for fp16 with fp32 accumulation
// - CANN auto-fusion may need tuning - potentially disable via MS_ENABLE_ACLNN=1 if crashes occur
//
// ACLNN Two-Phase Pattern:
// Every ACLNN operator uses:
// 1. Phase 1: Call xxxGetWorkspaceSize() to get workspaceSize and aclOpExecutor*
// 2. Phase 2: Call xxx(workspace, workspaceSize, executor, stream)
//---------------------------------------------------------------------------------

// Error checking macro for AscendCL
#define ACL_CHECK(call, name) \
  do { \
    aclError err = call; \
    if(err != ACL_SUCCESS) { \
      throw StringError(string(name) + " failed with ACL error: " + to_string((int)err)); \
    } \
  } while(0)

// Error checking macro for ACLNN operators
#define ACLNN_CHECK(call, name) \
  do { \
    aclnnStatus err = call; \
    if(err != ACLNN_SUCCESS) { \
      throw StringError(string(name) + " failed with ACLNN error: " + to_string((int)err)); \
    } \
  } while(0)

// cubeMathType: 0=KEEP_DTYPE, 1=ALLOW_FP32_DOWN_PRECISION, 2=USE_FP16, 3=USE_HF32
// Ascend 910ProA Cube Unit only supports FP16, so we must use 1 to allow
// automatic FP32->FP16 downcast for computation.
static const int8_t ASCEND_CUBE_MATH_TYPE = 1;  // ALLOW_FP32_DOWN_PRECISION

//---------------------------------------------------------------------------------
// AscendCL Helper Functions
//---------------------------------------------------------------------------------

// Helper to create an aclTensor from raw device pointer
// shape is in NCHW format (batch, channels, height, width) or (batch, channels) for 2D
// CANN 8.3 aclCreateTensor signature (9 parameters):
//   aclCreateTensor(viewDims, viewDimsNum, dataType, strides, storageOffset,
//                   format, storageDims, storageDimsNum, data)
static aclTensor* createAclTensor(
  void* data,
  const vector<int64_t>& shape,
  aclDataType dtype,
  aclFormat format
) {
  if(shape.empty()) {
    return nullptr;
  }

  // Compute contiguous strides
  vector<int64_t> strides(shape.size());
  strides[shape.size() - 1] = 1;
  for(int i = (int)shape.size() - 2; i >= 0; i--) {
    strides[i] = strides[i + 1] * shape[i + 1];
  }

  aclTensor* tensor = aclCreateTensor(
    shape.data(),                          // viewDims
    static_cast<uint64_t>(shape.size()),   // viewDimsNum
    dtype,                                 // dataType
    strides.data(),                        // strides
    (int64_t)0,                            // storageOffset
    format,                                // format
    shape.data(),                          // storageDims
    static_cast<uint64_t>(shape.size()),   // storageDimsNum
    data                                   // device data
  );

  return tensor;
}

// Helper to create a scalar aclScalar
static aclScalar* createAclScalar(float value, aclDataType dtype) {
  return aclCreateScalar(&value, dtype);
}

// Helper to create a float scalar with ACL_FLOAT type
static aclScalar* createFloatScalar(float value) {
  return aclCreateScalar(&value, ACL_FLOAT);
}


// Helper to destroy an aclTensor
static void destroyAclTensor(aclTensor* tensor) {
  if(tensor != nullptr) {
    aclDestroyTensor(tensor);
  }
}

// Helper to create aclIntArray
static aclIntArray* createAclIntArray(const vector<int64_t>& values) {
  return aclCreateIntArray(values.data(), static_cast<uint64_t>(values.size()));
}

//---------------------------------------------------------------------------------
// Memory Management Helpers
//---------------------------------------------------------------------------------

// Allocate device memory
static void* ascendMalloc(size_t size) {
  if(size == 0) return nullptr;
  void* ptr = nullptr;
  aclError ret = aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_HUGE_FIRST);
  if(ret != ACL_SUCCESS) {
    throw StringError("aclrtMalloc failed for size " + to_string(size) + " with error: " + to_string(ret));
  }
  return ptr;
}

// Free device memory
static void ascendFree(void* ptr) {
  if(ptr != nullptr) {
    aclrtFree(ptr);
  }
}

// Copy host to device (synchronous)
static void ascendCopyH2D(void* dst, const void* src, size_t size) {
  if(size == 0 || dst == nullptr || src == nullptr) return;
  aclError ret = aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE);
  if(ret != ACL_SUCCESS) {
    throw StringError("aclrtMemcpy H2D failed with error: " + to_string(ret));
  }
}

// Copy device to host (synchronous)
static void ascendCopyD2H(void* dst, const void* src, size_t size) {
  if(size == 0 || dst == nullptr || src == nullptr) return;
  aclError ret = aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST);
  if(ret != ACL_SUCCESS) {
    throw StringError("aclrtMemcpy D2H failed with error: " + to_string(ret));
  }
}

// Copy host to device (asynchronous)
static void ascendCopyH2DAsync(void* dst, const void* src, size_t size, aclrtStream stream) {
  if(size == 0 || dst == nullptr || src == nullptr) return;
  aclError ret = aclrtMemcpyAsync(dst, size, src, size, ACL_MEMCPY_HOST_TO_DEVICE, stream);
  if(ret != ACL_SUCCESS) {
    throw StringError("aclrtMemcpyAsync H2D failed with error: " + to_string(ret));
  }
}

// Copy device to host (asynchronous)
static void ascendCopyD2HAsync(void* dst, const void* src, size_t size, aclrtStream stream) {
  if(size == 0 || dst == nullptr || src == nullptr) return;
  aclError ret = aclrtMemcpyAsync(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_HOST, stream);
  if(ret != ACL_SUCCESS) {
    throw StringError("aclrtMemcpyAsync D2H failed with error: " + to_string(ret));
  }
}

// Copy device to device
static void ascendCopyD2D(void* dst, const void* src, size_t size) {
  if(size == 0 || dst == nullptr || src == nullptr) return;
  aclError ret = aclrtMemcpy(dst, size, src, size, ACL_MEMCPY_DEVICE_TO_DEVICE);
  if(ret != ACL_SUCCESS) {
    throw StringError("aclrtMemcpy D2D failed with error: " + to_string(ret));
  }
}

// Allocate and copy host data to device
static void* ascendMallocAndCopy(const void* hostData, size_t size) {
  if(size == 0 || hostData == nullptr) return nullptr;
  void* devicePtr = ascendMalloc(size);
  ascendCopyH2D(devicePtr, hostData, size);
  return devicePtr;
}

// Allocate device memory with FP16 conversion: convert float[] on host to aclFloat16[], then upload
// This is the key optimization for Ascend 910ProA - native FP16 weights eliminate per-op conversion
static void* ascendMallocAndCopyFP16(const float* hostData, size_t numElements) {
  if(numElements == 0 || hostData == nullptr) return nullptr;
  // Convert to FP16 on host using CANN's conversion function
  vector<aclFloat16> fp16Data(numElements);
  for(size_t i = 0; i < numElements; i++) {
    fp16Data[i] = aclFloatToFloat16(hostData[i]);
  }
  size_t fp16Bytes = numElements * sizeof(aclFloat16);
  void* devicePtr = ascendMalloc(fp16Bytes);
  ascendCopyH2D(devicePtr, fp16Data.data(), fp16Bytes);
  return devicePtr;
}

// Overload for vector<float>
static void* ascendMallocAndCopyFP16(const vector<float>& hostData) {
  return ascendMallocAndCopyFP16(hostData.data(), hostData.size());
}

// Device-side FP16 -> FP32 conversion (uses aclFloat16ToFloat from CANN 9.0)
// Does D2H + host convert + H2D round trip. Use for small tensors only.
static void castDeviceFP16ToFP32(aclrtStream stream, void* dstFP32, const void* srcFP16, size_t numElements) {
  size_t fp16Bytes = numElements * sizeof(aclFloat16);
  size_t fp32Bytes = numElements * sizeof(float);
  vector<aclFloat16> hostFP16(numElements);
  vector<float> hostFP32(numElements);
  aclrtMemcpy(hostFP16.data(), fp16Bytes, srcFP16, fp16Bytes, ACL_MEMCPY_DEVICE_TO_HOST);
  aclrtSynchronizeStream(stream);
  for(size_t i = 0; i < numElements; i++)
    hostFP32[i] = aclFloat16ToFloat(hostFP16[i]);
  aclrtMemcpy(dstFP32, fp32Bytes, hostFP32.data(), fp32Bytes, ACL_MEMCPY_HOST_TO_DEVICE);
}


//---------------------------------------------------------------------------------
// LoadedModel - simple wrapper around ModelDesc
//---------------------------------------------------------------------------------

struct LoadedModel {
  ModelDesc modelDesc;

  LoadedModel(const string& fileName, const string& expectedSha256) {
    ModelDesc::loadFromFileMaybeGZipped(fileName, modelDesc, expectedSha256);
  }

  LoadedModel() = delete;
  LoadedModel(const LoadedModel&) = delete;
  LoadedModel& operator=(const LoadedModel&) = delete;
};

LoadedModel* NeuralNet::loadModelFile(const string& file, const string& expectedSha256) {
  LoadedModel* loadedModel = new LoadedModel(file, expectedSha256);
  return loadedModel;
}

void NeuralNet::freeLoadedModel(LoadedModel* loadedModel) {
  delete loadedModel;
}

const ModelDesc& NeuralNet::getModelDesc(const LoadedModel* loadedModel) {
  return loadedModel->modelDesc;
}

//---------------------------------------------------------------------------------
// Global Initialization/Cleanup
//---------------------------------------------------------------------------------

static bool g_aclInitialized = false;

void NeuralNet::globalInitialize() {
  if(!g_aclInitialized) {
    aclError ret = aclInit(nullptr);
    if(ret != ACL_SUCCESS) {
      throw StringError("aclInit failed with error code: " + to_string(ret));
    }
    g_aclInitialized = true;
  }
}

void NeuralNet::globalCleanup() {
  if(g_aclInitialized) {
    aclFinalize();
    g_aclInitialized = false;
  }
}

void NeuralNet::printDevices() {
  uint32_t deviceCount = 0;
  aclError ret = aclrtGetDeviceCount(&deviceCount);
  if(ret != ACL_SUCCESS) {
    cout << "Failed to get Ascend NPU device count" << endl;
    return;
  }

  cout << "Found " << deviceCount << " Ascend NPU device(s)" << endl;
  for(uint32_t i = 0; i < deviceCount; i++) {
    // Note: aclrtGetSocName requires a device to be set first
    // For now, just print the index
    cout << "  Ascend NPU device " << i << endl;
  }
}

//---------------------------------------------------------------------------------
// Forward declarations
//---------------------------------------------------------------------------------

struct Model;
struct ScratchBuffers;
struct Buffers;

//---------------------------------------------------------------------------------
// ComputeContext - cross-thread NPU state
//---------------------------------------------------------------------------------

struct ComputeContext {
  int nnXLen;
  int nnYLen;
  enabled_t useFP16Mode;
  enabled_t useNHWCMode;
  vector<int> gpuIdxs;

  std::mutex cachedModelsMutex;
  std::map<std::string, std::shared_ptr<const Model>> cachedModels;
  std::map<std::string, int> cachedModelsRefCount;

  ComputeContext() = delete;
  ComputeContext(const ComputeContext&) = delete;
  ComputeContext& operator=(const ComputeContext&) = delete;

  ComputeContext(int nnX, int nnY, enabled_t fp16, enabled_t nhwc, const vector<int>& gpus)
    : nnXLen(nnX),
      nnYLen(nnY),
      useFP16Mode(fp16),
      useNHWCMode(nhwc),
      gpuIdxs(gpus)
  {}

  ~ComputeContext() {
    assert(cachedModels.size() == 0);
  }
};

//---------------------------------------------------------------------------------
// TensorCache - caches aclTensor descriptors per ComputeHandle
//---------------------------------------------------------------------------------

struct TensorCacheKey {
  void* data;
  int64_t dims[4];  // shape padded with 0s for unused dims
  int ndim;
  aclDataType dtype;
  aclFormat format;

  bool operator==(const TensorCacheKey& o) const {
    return data == o.data && ndim == o.ndim && dtype == o.dtype
      && format == o.format && memcmp(dims, o.dims, sizeof(dims)) == 0;
  }
};

struct TensorCacheKeyHash {
  size_t operator()(const TensorCacheKey& k) const {
    size_t h = std::hash<void*>()(k.data);
    for(int i = 0; i < k.ndim; i++)
      h ^= std::hash<int64_t>()(k.dims[i]) << (i + 1);
    h ^= std::hash<int>()(k.dtype) << 5;
    h ^= std::hash<int>()(k.format) << 7;
    return h;
  }
};

class TensorCache {
  std::unordered_map<TensorCacheKey, aclTensor*, TensorCacheKeyHash> cache;
public:
  aclTensor* get(void* data, const vector<int64_t>& shape,
                  aclDataType dtype, aclFormat format) {
    TensorCacheKey key;
    key.data = data;
    memset(key.dims, 0, sizeof(key.dims));
    key.ndim = (int)shape.size();
    for(int i = 0; i < key.ndim && i < 4; i++)
      key.dims[i] = shape[i];
    key.dtype = dtype;
    key.format = format;

    auto it = cache.find(key);
    if(it != cache.end()) return it->second;

    aclTensor* t = createAclTensor(data, shape, dtype, format);
    if(t == nullptr) {
      throw StringError("aclCreateTensor failed for shape [" +
        [&shape]() { string s; for(auto d : shape) s += to_string(d) + ","; return s; }()
        + "] dtype=" + to_string(dtype) + " format=" + to_string(format));
    }
    cache[key] = t;
    return t;
  }

  ~TensorCache() {
    for(auto& p : cache) {
      destroyAclTensor(p.second);
    }
  }
};

//---------------------------------------------------------------------------------
// ComputeHandle - per-thread handle
//---------------------------------------------------------------------------------

struct ComputeHandle {
  int deviceIdx;
  aclrtStream stream;

  const Model* model;
  ScratchBuffers* scratch;
  Buffers* buffers;

  bool usingFP16;
  int nnXLen;
  int nnYLen;
  bool requireExactNNLen;
  bool inputsUseNHWC;

  // Cached tensor descriptors - eliminates per-eval create/destroy overhead
  TensorCache tensorCache;

  // Cached alpha=1.0 scalar for Add operations
  aclScalar* alphaOneScalar;

  // Ping-pong buffer management for residual blocks
  // Instead of D2D copy, swap trunk/scratch pointers
  bool trunkIsPrimary;  // true = trunk is primaryBuf, scratch is secondaryBuf

  ComputeHandle() = delete;
  ComputeHandle(const ComputeHandle&) = delete;
  ComputeHandle& operator=(const ComputeHandle&) = delete;

  ComputeHandle(int device, bool fp16, int nnX, int nnY, bool exactLen, bool nhwc)
    : deviceIdx(device),
      stream(nullptr),
      model(nullptr),
      scratch(nullptr),
      buffers(nullptr),
      usingFP16(fp16),
      nnXLen(nnX),
      nnYLen(nnY),
      requireExactNNLen(exactLen),
      inputsUseNHWC(nhwc),
      alphaOneScalar(nullptr),
      trunkIsPrimary(true)
  {
  }

  ~ComputeHandle() {
    // Set device context first since destructor may run on a different thread
    // CANN's device binding is thread-local
    aclrtSetDevice(deviceIdx);

    if(alphaOneScalar != nullptr) {
      aclDestroyScalar(alphaOneScalar);
    }
    if(stream != nullptr) {
      aclrtDestroyStream(stream);
    }
    aclrtResetDevice(deviceIdx);
  }

  void initScalars() {
    if(alphaOneScalar == nullptr) {
      alphaOneScalar = createFloatScalar(1.0f);
    }
  }
};

//---------------------------------------------------------------------------------
// InputBuffers - host-side buffers
//---------------------------------------------------------------------------------

struct InputBuffers {
  int maxBatchSize;

  size_t singleInputElts;
  size_t singleInputBytes;
  size_t singleInputGlobalElts;
  size_t singleInputGlobalBytes;
  size_t singleInputMetaElts;
  size_t singleInputMetaBytes;

  size_t singlePolicyPassResultElts;
  size_t singlePolicyPassResultBytes;
  size_t singlePolicyResultElts;
  size_t singlePolicyResultBytes;
  size_t singleValueResultElts;
  size_t singleValueResultBytes;
  size_t singleScoreValueResultElts;
  size_t singleScoreValueResultBytes;
  size_t singleOwnershipResultElts;
  size_t singleOwnershipResultBytes;

  size_t userInputBufferBytes;
  size_t userInputGlobalBufferBytes;
  size_t userInputMetaBufferBytes;
  size_t policyPassResultBufferBytes;
  size_t policyResultBufferBytes;
  size_t valueResultBufferBytes;
  size_t scoreValueResultBufferBytes;
  size_t ownershipResultBufferBytes;

  float* userInputBuffer;
  float* userInputGlobalBuffer;
  float* userInputMetaBuffer;

  float* policyPassResults;
  float* policyResults;
  float* valueResults;
  float* scoreValueResults;
  float* ownershipResults;

  InputBuffers(const LoadedModel* loadedModel, int maxBatchSz, int nnX, int nnY) {
    const ModelDesc& m = loadedModel->modelDesc;

    maxBatchSize = maxBatchSz;
    singleInputElts = (size_t)m.numInputChannels * nnX * nnY;
    singleInputBytes = singleInputElts * sizeof(float);
    singleInputGlobalElts = (size_t)m.numInputGlobalChannels;
    singleInputGlobalBytes = singleInputGlobalElts * sizeof(float);
    singleInputMetaElts = (size_t)m.numInputMetaChannels;
    singleInputMetaBytes = singleInputMetaElts * sizeof(float);

    singlePolicyPassResultElts = (size_t)m.numPolicyChannels;
    singlePolicyPassResultBytes = singlePolicyPassResultElts * sizeof(float);
    singlePolicyResultElts = (size_t)m.numPolicyChannels * nnX * nnY;
    singlePolicyResultBytes = singlePolicyResultElts * sizeof(float);
    singleValueResultElts = (size_t)m.numValueChannels;
    singleValueResultBytes = singleValueResultElts * sizeof(float);
    singleScoreValueResultElts = (size_t)m.numScoreValueChannels;
    singleScoreValueResultBytes = singleScoreValueResultElts * sizeof(float);
    singleOwnershipResultElts = (size_t)m.numOwnershipChannels * nnX * nnY;
    singleOwnershipResultBytes = singleOwnershipResultElts * sizeof(float);

    userInputBufferBytes = (size_t)m.numInputChannels * maxBatchSz * nnX * nnY * sizeof(float);
    userInputGlobalBufferBytes = (size_t)m.numInputGlobalChannels * maxBatchSz * sizeof(float);
    userInputMetaBufferBytes = (size_t)m.numInputMetaChannels * maxBatchSz * sizeof(float);
    policyPassResultBufferBytes = (size_t)maxBatchSz * m.numPolicyChannels * sizeof(float);
    policyResultBufferBytes = (size_t)maxBatchSz * m.numPolicyChannels * nnX * nnY * sizeof(float);
    valueResultBufferBytes = (size_t)maxBatchSz * m.numValueChannels * sizeof(float);
    scoreValueResultBufferBytes = (size_t)maxBatchSz * m.numScoreValueChannels * sizeof(float);
    ownershipResultBufferBytes = (size_t)maxBatchSz * nnX * nnY * m.numOwnershipChannels * sizeof(float);

    userInputBuffer = new float[singleInputElts * maxBatchSz];
    userInputGlobalBuffer = new float[singleInputGlobalElts * maxBatchSz];
    if(m.numInputMetaChannels > 0) {
      userInputMetaBuffer = new float[singleInputMetaElts * maxBatchSz];
    } else {
      userInputMetaBuffer = nullptr;
    }

    policyPassResults = new float[singlePolicyPassResultElts * maxBatchSz];
    policyResults = new float[singlePolicyResultElts * maxBatchSz];
    valueResults = new float[singleValueResultElts * maxBatchSz];
    scoreValueResults = new float[singleScoreValueResultElts * maxBatchSz];
    ownershipResults = new float[singleOwnershipResultElts * maxBatchSz];
  }

  ~InputBuffers() {
    delete[] userInputBuffer;
    delete[] userInputGlobalBuffer;
    if(userInputMetaBuffer != nullptr) {
      delete[] userInputMetaBuffer;
    }
    delete[] policyPassResults;
    delete[] policyResults;
    delete[] valueResults;
    delete[] scoreValueResults;
    delete[] ownershipResults;
  }

  InputBuffers() = delete;
  InputBuffers(const InputBuffers&) = delete;
  InputBuffers& operator=(const InputBuffers&) = delete;
};

InputBuffers* NeuralNet::createInputBuffers(const LoadedModel* loadedModel, int maxBatchSize, int nnXLen, int nnYLen) {
  return new InputBuffers(loadedModel, maxBatchSize, nnXLen, nnYLen);
}

void NeuralNet::freeInputBuffers(InputBuffers* inputBuffers) {
  delete inputBuffers;
}

//---------------------------------------------------------------------------------
// ScratchBuffers - workspace allocator
//---------------------------------------------------------------------------------

struct ScratchBuffers {
  const size_t batchXYFloatBytes;
  const size_t batchFloatBytes;
  const size_t batchXYBytes;
  const size_t batchBytes;

  const int maxBatchSize;
  const int nnXLen;
  const int nnYLen;
  const bool useFP16;

  // Pre-allocated workspace for ACLNN operations
  void* workspaceBuf;
  size_t workspaceBytes;

  // Track allocated buffers for cleanup
  vector<void*> allocatedBuffers;

  ScratchBuffers() = delete;
  ScratchBuffers(const ScratchBuffers&) = delete;
  ScratchBuffers& operator=(const ScratchBuffers&) = delete;

  ScratchBuffers(int maxBatchSz, int nnX, int nnY, bool fp16, size_t maxWorkspaceNeeded)
    : batchXYFloatBytes((size_t)maxBatchSz * nnX * nnY * sizeof(float)),
      batchFloatBytes((size_t)maxBatchSz * sizeof(float)),
      batchXYBytes((size_t)maxBatchSz * nnX * nnY * (fp16 ? sizeof(aclFloat16) : sizeof(float))),
      batchBytes((size_t)maxBatchSz * (fp16 ? sizeof(aclFloat16) : sizeof(float))),
      maxBatchSize(maxBatchSz),
      nnXLen(nnX),
      nnYLen(nnY),
      useFP16(fp16),
      workspaceBuf(nullptr),
      workspaceBytes(0)
  {
    // Pre-allocate workspace
    workspaceBytes = maxWorkspaceNeeded;
    if(workspaceBytes > 0) {
      workspaceBuf = ascendMalloc(workspaceBytes);
    }
  }

  ~ScratchBuffers() {
    // Free any allocated buffers
    for(void* buf : allocatedBuffers) {
      ascendFree(buf);
    }
    if(workspaceBuf != nullptr) {
      ascendFree(workspaceBuf);
    }
  }

  // Allocate a buffer of the given size and track it for cleanup
  void* allocate(size_t size) {
    void* buf = ascendMalloc(size);
    allocatedBuffers.push_back(buf);
    return buf;
  }

  // Release tracking for a buffer (does not free immediately, just removes from tracking)
  void release(void* buf) {
    // Find and remove from tracking (buffer will be freed in destructor)
    for(auto it = allocatedBuffers.begin(); it != allocatedBuffers.end(); ++it) {
      if(*it == buf) {
        allocatedBuffers.erase(it);
        ascendFree(buf);
        return;
      }
    }
  }

  size_t getBufSizeXY(int channels) const {
    return channels * batchXYBytes;
  }
  size_t getBufSizeXYFloat(int channels) const {
    return channels * batchXYFloatBytes;
  }
  size_t getBufSizeFloat(int channels) const {
    return channels * batchFloatBytes;
  }
  size_t getBufSize(int channels) const {
    return channels * batchBytes;
  }
};

//---------------------------------------------------------------------------------
// Buffers - device-side buffers
//---------------------------------------------------------------------------------

struct Buffers {
  // Input buffers (device)
  void* inputBuf;           // Spatial input (NCHW)
  void* inputGlobalBuf;     // Global features (NC)
  void* inputMetaBuf;       // Meta features (NC, optional)

  // For FP16 mode, we also need float versions for initial copy
  void* inputBufFloat;
  void* inputGlobalBufFloat;
  void* inputMetaBufFloat;

  // Output buffers (device, always float32 for final output)
  float* policyPassBuf;
  float* policyBuf;
  float* valueBuf;
  float* scoreValueBuf;
  void* ownershipBuf;

  // Workspace
  void* workspaceBuf;
  size_t workspaceBytes;

  // Pre-allocated intermediate buffers (eliminates per-eval malloc/free)
  void* trunkBuf;           // Trunk output buffer
  void* maskBuf;            // Mask buffer
  void* scratchBuf;         // Scratch buffer for residual blocks
  void* residualMidBuf;     // Mid buffer for residual blocks (avoids in-place conv)
  void* p1OutBuf;           // Policy head intermediate (ALWAYS float - CUDA: "Need to hold floats, not just halfs")
  void* p1Out2Buf;          // Policy head p1Out converted to float in FP16 mode (for bias add)
  void* g1OutBuf;           // Policy head gpool intermediate
  void* g1Out2Buf;          // Policy head gpool intermediate 2
  void* g1ConcatBuf;        // Policy head gpool concat (always FP32)
  void* g1BiasBuf;          // Policy head bias (always FP32 - gpoolToBiasMul uses FP32)
  void* p1PassBuf;          // Policy pass intermediate (always FP32)
  void* v1OutBuf;           // Value head intermediate
  void* v1Out2Buf;          // Value head intermediate 2
  void* v1MeanBuf;          // Value head mean buffer (always FP32)
  void* v2OutBuf;           // Value head v2 output (always FP32)
  void* ownershipScratchBuf; // Ownership scratch buffer (FP16 conv output before FP32 conversion)
  void* maskSumBuf;         // Mask sum buffer (float, count of valid positions per batch)
  void* maskFloatBuf;       // Mask as float (for BN operations)

  // Global pooling block buffers (needed by GlobalPoolingResidualBlock in trunk)
  void* gpoolConvOutBuf;      // Gpool conv output (full spatial: [B, gpoolC, H, W])
  void* gpoolScratchBuf;      // Gpool pooling intermediates (mean, max, concat, bias)
  size_t gpoolConvOutBufBytes;
  size_t gpoolScratchBufBytes;

  // Size tracking
  size_t inputBufBytes;
  size_t inputGlobalBufBytes;
  size_t inputMetaBufBytes;
  size_t inputBufBytesFloat;
  size_t inputGlobalBufBytesFloat;
  size_t inputMetaBufBytesFloat;
  size_t policyPassBufBytes;
  size_t policyBufBytes;
  size_t valueBufBytes;
  size_t scoreValueBufBytes;
  size_t ownershipBufBytes;
  size_t trunkBufBytes;
  size_t maskBufBytes;
  size_t scratchBufBytes;

  Buffers(const ModelDesc& m, int maxBatchSize, int nnXLen, int nnYLen, bool useFP16, size_t extraWorkspace)
    : inputBuf(nullptr),
      inputGlobalBuf(nullptr),
      inputMetaBuf(nullptr),
      inputBufFloat(nullptr),
      inputGlobalBufFloat(nullptr),
      inputMetaBufFloat(nullptr),
      policyPassBuf(nullptr),
      policyBuf(nullptr),
      valueBuf(nullptr),
      scoreValueBuf(nullptr),
      ownershipBuf(nullptr),
      workspaceBuf(nullptr),
      workspaceBytes(0),
      trunkBuf(nullptr),
      maskBuf(nullptr),
      scratchBuf(nullptr),
      p1OutBuf(nullptr),
      p1Out2Buf(nullptr),
      g1OutBuf(nullptr),
      g1Out2Buf(nullptr),
      g1ConcatBuf(nullptr),
      g1BiasBuf(nullptr),
      p1PassBuf(nullptr),
      v1OutBuf(nullptr),
      v1Out2Buf(nullptr),
      v1MeanBuf(nullptr),
      v2OutBuf(nullptr),
      ownershipScratchBuf(nullptr),
      maskSumBuf(nullptr),
      maskFloatBuf(nullptr),
      gpoolConvOutBuf(nullptr),
      gpoolScratchBuf(nullptr),
      gpoolConvOutBufBytes(0),
      gpoolScratchBufBytes(0)
  {
    size_t eltSize = useFP16 ? sizeof(aclFloat16) : sizeof(float);

    inputBufBytes = (size_t)m.numInputChannels * maxBatchSize * nnXLen * nnYLen * eltSize;
    inputGlobalBufBytes = (size_t)m.numInputGlobalChannels * maxBatchSize * eltSize;
    inputMetaBufBytes = (size_t)m.numInputMetaChannels * maxBatchSize * eltSize;

    inputBufBytesFloat = (size_t)m.numInputChannels * maxBatchSize * nnXLen * nnYLen * sizeof(float);
    inputGlobalBufBytesFloat = (size_t)m.numInputGlobalChannels * maxBatchSize * sizeof(float);
    inputMetaBufBytesFloat = (size_t)m.numInputMetaChannels * maxBatchSize * sizeof(float);

    policyPassBufBytes = (size_t)maxBatchSize * m.numPolicyChannels * sizeof(float);
    policyBufBytes = (size_t)maxBatchSize * m.numPolicyChannels * nnXLen * nnYLen * sizeof(float);
    valueBufBytes = (size_t)maxBatchSize * m.numValueChannels * sizeof(float);
    scoreValueBufBytes = (size_t)maxBatchSize * m.numScoreValueChannels * sizeof(float);
    ownershipBufBytes = (size_t)maxBatchSize * nnXLen * nnYLen * m.numOwnershipChannels * sizeof(float);

    // Intermediate buffer sizes
    trunkBufBytes = (size_t)m.trunk.trunkNumChannels * maxBatchSize * nnXLen * nnYLen * eltSize;
    maskBufBytes = (size_t)1 * maxBatchSize * nnXLen * nnYLen * eltSize;
    scratchBufBytes = trunkBufBytes;

    // Allocate input buffers
    inputBuf = ascendMalloc(inputBufBytes);
    inputGlobalBuf = ascendMalloc(inputGlobalBufBytes);
    if(m.numInputMetaChannels > 0) {
      inputMetaBuf = ascendMalloc(inputMetaBufBytes);
    }

    // For FP16 mode, allocate float buffers for initial host copy
    if(useFP16) {
      inputBufFloat = ascendMalloc(inputBufBytesFloat);
      inputGlobalBufFloat = ascendMalloc(inputGlobalBufBytesFloat);
      if(m.numInputMetaChannels > 0) {
        inputMetaBufFloat = ascendMalloc(inputMetaBufBytesFloat);
      }
    }

    // Allocate output buffers (always float32)
    policyPassBuf = (float*)ascendMalloc(policyPassBufBytes);
    policyBuf = (float*)ascendMalloc(policyBufBytes);
    valueBuf = (float*)ascendMalloc(valueBufBytes);
    scoreValueBuf = (float*)ascendMalloc(scoreValueBufBytes);
    ownershipBuf = ascendMalloc(ownershipBufBytes);

    // Allocate intermediate buffers
    trunkBuf = ascendMalloc(trunkBufBytes);
    maskBuf = ascendMalloc(maskBufBytes);
    scratchBuf = ascendMalloc(scratchBufBytes);
    residualMidBuf = ascendMalloc(scratchBufBytes);  // Same size as trunk

    // Policy head intermediate buffers
    // CRITICAL: Match CUDA buffer sizes exactly
    int p1Channels = m.policyHead.p1Conv.outChannels;
    int g1Channels = m.policyHead.g1Conv.outChannels;
    // p1Out: ALWAYS float-sized (CUDA comment: "Need to hold floats, not just halfs")
    p1OutBuf = ascendMalloc((size_t)p1Channels * maxBatchSize * nnXLen * nnYLen * sizeof(float));
    // p1Out2: float buffer for FP16->FP32 conversion of p1Out before bias add (only in FP16 mode)
    if(useFP16) {
      p1Out2Buf = ascendMalloc((size_t)p1Channels * maxBatchSize * nnXLen * nnYLen * sizeof(float));
    }
    // g1Out, g1Out2: FP16 or FP32 depending on mode (g1Conv and g1BN use FP16)
    g1OutBuf = ascendMalloc((size_t)g1Channels * maxBatchSize * nnXLen * nnYLen * eltSize);
    g1Out2Buf = ascendMalloc((size_t)g1Channels * maxBatchSize * nnXLen * nnYLen * eltSize);
    // g1Concat: ALWAYS float (global pooling always outputs float)
    g1ConcatBuf = ascendMalloc((size_t)g1Channels * 3 * maxBatchSize * sizeof(float));
    // g1Bias: ALWAYS float (gpoolToBiasMul uses FP32)
    g1BiasBuf = ascendMalloc((size_t)p1Channels * maxBatchSize * sizeof(float));
    // p1PassBuf: ALWAYS float (gpoolToPassMul uses FP32)
    p1PassBuf = ascendMalloc((size_t)p1Channels * maxBatchSize * sizeof(float));

    // Value head intermediate buffers
    // CRITICAL: Match CUDA buffer sizes exactly
    int v1Channels = m.valueHead.v1Conv.outChannels;
    int v2Channels = m.valueHead.v2Mul.outChannels;
    int ownershipChannels = m.valueHead.vOwnershipConv.outChannels;
    v1OutBuf = ascendMalloc((size_t)v1Channels * maxBatchSize * nnXLen * nnYLen * eltSize);
    v1Out2Buf = ascendMalloc((size_t)v1Channels * maxBatchSize * nnXLen * nnYLen * eltSize);
    // v1Mean: ALWAYS float
    v1MeanBuf = ascendMalloc((size_t)v1Channels * 3 * maxBatchSize * sizeof(float));
    // v2Out: ALWAYS float (v2Mul uses FP32)
    v2OutBuf = ascendMalloc((size_t)v2Channels * maxBatchSize * sizeof(float));
    // ownershipScratchBuf: only needed in FP16 mode for conv output before FP32 conversion
    if(useFP16) {
      ownershipScratchBuf = ascendMalloc((size_t)ownershipChannels * maxBatchSize * nnXLen * nnYLen * sizeof(aclFloat16));
    }

    // Mask buffers
    maskBuf = ascendMalloc(maskBufBytes);
    // maskFloatBuf: ALWAYS float (needed for BN operations and pooling)
    maskFloatBuf = ascendMalloc((size_t)maxBatchSize * nnXLen * nnYLen * sizeof(float));
    // maskSumBuf: ALWAYS float (count of valid board positions per batch)
    maskSumBuf = ascendMalloc((size_t)maxBatchSize * sizeof(float));

    // Allocate workspace
    workspaceBytes = extraWorkspace;
    if(workspaceBytes > 0) {
      workspaceBuf = ascendMalloc(workspaceBytes);
    }

    // Allocate global pooling block buffers (if model has gpool blocks)
    int maxGpoolChannels = 0;
    int maxGpoolRegularChannels = 0;
    for(const auto& blockPair : m.trunk.blocks) {
      if(blockPair.first == GLOBAL_POOLING_BLOCK_KIND) {
        const auto* desc = static_cast<const GlobalPoolingResidualBlockDesc*>(blockPair.second.get());
        maxGpoolChannels = std::max(maxGpoolChannels, desc->gpoolConv.outChannels);
        maxGpoolRegularChannels = std::max(maxGpoolRegularChannels, desc->regularConv.outChannels);
      }
    }
    if(maxGpoolChannels > 0) {
      // gpoolConvOutBuf: full spatial output of gpoolConv [B, gpoolC, H, W]
      gpoolConvOutBufBytes = (size_t)maxBatchSize * maxGpoolChannels * nnXLen * nnYLen * eltSize;
      gpoolConvOutBuf = ascendMalloc(gpoolConvOutBufBytes);

      // gpoolScratchBuf: pooling intermediates + concat + bias
      // Layout: mean(dtype) + meanFP32 + max(dtype) + maxFP32 + scaledMean(FP32)
      //         + concat(FP32) + bias(FP32)
      size_t poolElts = (size_t)maxBatchSize * maxGpoolChannels;
      size_t pBytesDtype = poolElts * (useFP16 ? sizeof(aclFloat16) : sizeof(float));
      size_t pBytesFP32 = poolElts * sizeof(float);
      gpoolScratchBufBytes = 2 * pBytesDtype + 5 * pBytesFP32
                           + poolElts * 3 * sizeof(float)  // gpoolConcat
                           + (size_t)maxBatchSize * maxGpoolRegularChannels * sizeof(float);  // gpoolBias
      gpoolScratchBuf = ascendMalloc(gpoolScratchBufBytes);
    }
  }

  ~Buffers() {
    ascendFree(inputBuf);
    ascendFree(inputGlobalBuf);
    ascendFree(inputMetaBuf);
    ascendFree(inputBufFloat);
    ascendFree(inputGlobalBufFloat);
    ascendFree(inputMetaBufFloat);
    ascendFree(policyPassBuf);
    ascendFree(policyBuf);
    ascendFree(valueBuf);
    ascendFree(scoreValueBuf);
    ascendFree(ownershipBuf);
    ascendFree(workspaceBuf);
    // Free intermediate buffers
    ascendFree(trunkBuf);
    ascendFree(maskBuf);
    ascendFree(scratchBuf);
    ascendFree(residualMidBuf);
    ascendFree(p1OutBuf);
    ascendFree(p1Out2Buf);
    ascendFree(g1OutBuf);
    ascendFree(g1Out2Buf);
    ascendFree(g1ConcatBuf);
    ascendFree(g1BiasBuf);
    ascendFree(p1PassBuf);
    ascendFree(v1OutBuf);
    ascendFree(v1Out2Buf);
    ascendFree(v1MeanBuf);
    ascendFree(v2OutBuf);
    ascendFree(ownershipScratchBuf);
    // Free mask buffers
    ascendFree(maskFloatBuf);
    ascendFree(maskSumBuf);
    // Free gpool block buffers
    ascendFree(gpoolConvOutBuf);
    ascendFree(gpoolScratchBuf);
  }

  Buffers() = delete;
  Buffers(const Buffers&) = delete;
  Buffers& operator=(const Buffers&) = delete;
};

//---------------------------------------------------------------------------------
// Basic Layer Implementations
//---------------------------------------------------------------------------------

// ConvLayer - convolution layer using ACLNN
struct ConvLayer {
  const string name;
  const int inChannels;
  const int outChannels;
  const int convYSize;
  const int convXSize;
  const int dilationY;
  const int dilationX;

  void* filterBuf;              // Device memory for weights (NCHW: outC, inC, H, W)
  aclDataType dtype;
  bool useFP16;
  int8_t cubeMathType;          // 0=KEEP_DTYPE (native FP16), 1=ALLOW_FP32_DOWN_PRECISION

  // Cached aclIntArray objects - created once in constructor, reused every apply()
  aclIntArray* stridesArr;
  aclIntArray* paddingsArr;
  aclIntArray* dilationsArr;
  aclIntArray* outputPaddingArr;

  ConvLayer() = delete;
  ConvLayer(const ConvLayer&) = delete;
  ConvLayer& operator=(const ConvLayer&) = delete;

  ConvLayer(const ConvLayerDesc* desc, bool useFP16_)
    : name(desc->name),
      inChannels(desc->inChannels),
      outChannels(desc->outChannels),
      convYSize(desc->convYSize),
      convXSize(desc->convXSize),
      dilationY(desc->dilationY),
      dilationX(desc->dilationX),
      useFP16(useFP16_),
      stridesArr(nullptr),
      paddingsArr(nullptr),
      dilationsArr(nullptr),
      outputPaddingArr(nullptr)
  {
    // Allocate and copy weights to device with native FP16 conversion
    // KataGo weights are in (outC, inC, H, W) format which is NCHW-compatible
    if(useFP16) {
      filterBuf = ascendMallocAndCopyFP16(desc->weights.data(), desc->weights.size());
      dtype = ACL_FLOAT16;
      cubeMathType = 1;  // ALLOW_FP32_DOWN_PRECISION - more compatible than KEEP_DTYPE
    } else {
      size_t weightBytes = desc->weights.size() * sizeof(float);
      filterBuf = ascendMallocAndCopy(desc->weights.data(), weightBytes);
      dtype = ACL_FLOAT;
      cubeMathType = 1;  // ALLOW_FP32_DOWN_PRECISION - let CANN convert FP32->FP16
    }

    // Pre-create cached aclIntArray objects for convolution parameters
    // These are constant for this layer and reused every apply() call
    int paddingY = (convYSize / 2) * dilationY;
    int paddingX = (convXSize / 2) * dilationX;

    stridesArr = createAclIntArray({1, 1});
    paddingsArr = createAclIntArray({paddingY, paddingX});
    dilationsArr = createAclIntArray({dilationY, dilationX});
    outputPaddingArr = createAclIntArray({0, 0});
  }

  ~ConvLayer() {
    ascendFree(filterBuf);
    // Destroy cached aclIntArray objects
    if(stridesArr) aclDestroyIntArray(stridesArr);
    if(paddingsArr) aclDestroyIntArray(paddingsArr);
    if(dilationsArr) aclDestroyIntArray(dilationsArr);
    if(outputPaddingArr) aclDestroyIntArray(outputPaddingArr);
  }

  size_t requiredWorkspaceBytes(int batchSize, int nnXLen, int nnYLen, aclrtStream stream) const {
    // Query ACLNN for workspace size
    // Create dummy tensors to query
    vector<int64_t> inputShape = {batchSize, inChannels, nnYLen, nnXLen};
    vector<int64_t> outputShape = {batchSize, outChannels, nnYLen, nnXLen};
    vector<int64_t> weightShape = {outChannels, inChannels, convYSize, convXSize};

    // Create tensors (with nullptr data for size query)
    aclTensor* inputTensor = createAclTensor(nullptr, inputShape, dtype, ACL_FORMAT_NCHW);
    aclTensor* weightTensor = createAclTensor(nullptr, weightShape, dtype, ACL_FORMAT_NCHW);
    aclTensor* outputTensor = createAclTensor(nullptr, outputShape, dtype, ACL_FORMAT_NCHW);

    // Use cached aclIntArray objects created in constructor
    uint64_t workspaceSize = 0;
    aclOpExecutor* executor = nullptr;

    // CANN 8.3 signature:
    // aclnnConvolutionGetWorkspaceSize(input, weight, bias, stride, padding, dilation,
    //                                   transposed, outputPadding, groups, output,
    //                                   cubeMathType, workspaceSize, executor)
    aclnnStatus status = aclnnConvolutionGetWorkspaceSize(
      inputTensor,
      weightTensor,
      nullptr,        // bias
      stridesArr,
      paddingsArr,
      dilationsArr,
      false,          // transposed
      outputPaddingArr,
      (int64_t)1,     // groups
      outputTensor,
      cubeMathType,      // cubeMathType: 0 for native FP16, 1 for FP32->FP16 conversion
      &workspaceSize,
      &executor
    );

    // Cleanup tensors only (cached arrays are destroyed in destructor)
    destroyAclTensor(inputTensor);
    destroyAclTensor(weightTensor);
    destroyAclTensor(outputTensor);

    if(status != ACLNN_SUCCESS) {
      // Return a conservative estimate
      return 1024 * 1024 * 16; // 16 MB fallback
    }

    (void)stream;
    return workspaceSize;
  }

  // Check if this is a 1x1 convolution that should use MatMul path
  // DaVinci Cube requires 16x16 blocks, so outChannels < 16 fails with aclnnConvolution
  bool shouldUseMatMulPath() const {
    return convYSize == 1 && convXSize == 1 && dilationY == 1 && dilationX == 1 && outChannels < 16;
  }

  void apply(
    ComputeHandle* handle,
    aclrtStream stream,
    int batchSize,
    int nnXLen,
    int nnYLen,
    bool accumulate,
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    // For 1x1 convolutions with small output channels, use MatMul path
    // to avoid DaVinci Cube alignment issues (requires 16x16 blocks)
    if(shouldUseMatMulPath()) {
      applyViaMatMul(handle, stream, batchSize, nnXLen, nnYLen, accumulate, inputBuf, outputBuf, workspaceBuf, workspaceBytes);
      return;
    }

    // Standard convolution path for larger output channels
    applyViaConvolution(handle, stream, batchSize, nnXLen, nnYLen, accumulate, inputBuf, outputBuf, workspaceBuf, workspaceBytes);
  }

  void applyViaMatMul(
    ComputeHandle* handle,
    aclrtStream stream,
    int batchSize,
    int nnXLen,
    int nnYLen,
    bool accumulate,
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    // 1x1 convolution via MatMul:
    // Input: [N, C_in, H, W] -> reshape to [N*H*W, C_in]
    // Weight: [C_out, C_in, 1, 1] -> reshape to [C_in, C_out] (transposed)
    // MatMul: [N*H*W, C_in] @ [C_in, C_out] = [N*H*W, C_out]
    // Output: [N*H*W, C_out] -> reshape to [N, C_out, H, W]

    int64_t N = batchSize;
    int64_t H = nnYLen;
    int64_t W = nnXLen;
    int64_t C_in = inChannels;
    int64_t C_out = outChannels;
    int64_t NHW = N * H * W;

    vector<aclTensor*> localTensors;

    // Input tensor: [N, C_in, H, W] viewed as [NHW, C_in] for matmul
    // We use ACL_FORMAT_ND (row-major) for 2D tensors
    vector<int64_t> input2DShape = {NHW, C_in};
    aclTensor* input2DTensor;
    if(handle != nullptr) {
      input2DTensor = handle->tensorCache.get(inputBuf, input2DShape, dtype, ACL_FORMAT_ND);
    } else {
      input2DTensor = createAclTensor(inputBuf, input2DShape, dtype, ACL_FORMAT_ND);
      localTensors.push_back(input2DTensor);
    }

    // Weight tensor: [C_out, C_in, 1, 1] viewed as [C_out, C_in] for matmul
    // For matmul: input @ weight.T, so we need weight shape [C_in, C_out]
    // But weights are stored as [C_out, C_in], so we create tensor with that shape
    // and use aclnnMatmul which will compute input @ weight_transposed
    vector<int64_t> weight2DShape = {C_out, C_in};  // Weights stored as [C_out, C_in]
    aclTensor* weight2DTensor;
    if(handle != nullptr) {
      weight2DTensor = handle->tensorCache.get(filterBuf, weight2DShape, dtype, ACL_FORMAT_ND);
    } else {
      weight2DTensor = createAclTensor(filterBuf, weight2DShape, dtype, ACL_FORMAT_ND);
      localTensors.push_back(weight2DTensor);
    }

    // Output tensor: [N, C_out, H, W] viewed as [NHW, C_out]
    vector<int64_t> output2DShape = {NHW, C_out};
    aclTensor* output2DTensor;
    if(handle != nullptr) {
      output2DTensor = handle->tensorCache.get(outputBuf, output2DShape, dtype, ACL_FORMAT_ND);
    } else {
      output2DTensor = createAclTensor(outputBuf, output2DShape, dtype, ACL_FORMAT_ND);
      localTensors.push_back(output2DTensor);
    }

    // Use aclnnMm (matrix multiply) which does: output = input @ weight
    // But we need: output = input @ weight.T
    // aclnnMm doesn't support transpose, so we use aclnnMatmul
    // aclnnMatmul does: output = mat1 @ mat2 (also no transpose)
    // Solution: Create a view of weight with transposed dimensions

    // Actually, we need to transpose the weight. Let's use aclnnMatmul with
    // the correct tensor view. The weight is [C_out, C_in], we need [C_in, C_out].
    // We can achieve this by creating the weight tensor with transposed storage.

    // Alternative: Use aclnnLinear which does output = input @ weight.T + bias
    // This is exactly what we need! aclnnLinear(input, weight, bias) = input @ weight.T + bias

    // For now, let's manually do the transpose using reshape trick
    // Actually, let's use aclnnMatmulTransposed if available, or aclnnMm

    // Simplest approach: Use aclnnLinear which computes input @ weight.T + bias
    // aclnnLinearGetWorkspaceSize(input, weight, bias, output, ...)
    // This is perfect for our case!

    // Wait, aclnnLinear might not be available. Let's check what we have.
    // For now, let's use a different approach: manual transpose then matmul

    // Actually, the simplest fix is to use the existing weight tensor [C_out, C_in]
    // and compute input @ weight.T by reshaping the computation.

    // Since aclnnMatmul does: C = A @ B
    // If A = [NHW, C_in] and B = [C_in, C_out], then C = [NHW, C_out]
    // But our weight is stored as [C_out, C_in], not [C_in, C_out]

    // Solution: Swap dimensions when creating the weight tensor view
    // We create weight tensor with shape [C_in, C_out] by using different strides
    // This is essentially a transpose view.

    // Actually, let's just do: output = (input @ weight.T) = (weight @ input.T).T
    // That's more complex. Let's try a simpler approach.

    // NEW APPROACH: Use aclnnMm which does output = input @ weight
    // But create the weight tensor with shape [C_in, C_out] by interpreting
    // the data differently. Since weights are stored row-major as [C_out, C_in],
    // we can't just reinterpret. We need to actually transpose.

    // SIMPLEST FIX: Use aclnnMatMul with transposedB parameter
    // Check if aclnn supports this...

    // For now, let's use a workspace to transpose the weight first,
    // then do matmul.

    // Actually, looking at the CANN API, aclnnMatmul does support transposition!
    // Let me check the signature...

    // aclnnMatmulGetWorkspaceSize(mat1, mat2, output, cubeMathType, workspaceSize, executor)
    // This does: output = mat1 @ mat2 (no transpose option visible)

    // Let's try aclnnMm: output = input @ mat2
    // Same issue - no transpose.

    // OK, let's just do a manual approach:
    // 1. Transpose weight from [C_out, C_in] to [C_in, C_out] using aclnnCopy or similar
    // 2. Then do matmul

    // For simplicity, let's just store the weights in transposed form
    // for 1x1 convs with small output channels.

    // Actually, the EASIEST fix is to change how we store weights for these layers.
    // But that requires changes to the constructor.

    // For a quick fix, let's transpose at runtime using aclnnCopy with permute
    // But aclnnCopy doesn't support permute...

    // Let's check if aclnnTranspose is available...
    // It might be in aclnnop/aclnn_transpose.h or similar.

    // For now, let's try a different approach: use aclnnMm and accept that
    // the result will be wrong until we fix the transpose issue.
    // NO - that would give wrong results.

    // REAL FIX: We need to transpose the weight. Let me add a transposed weight buffer
    // for 1x1 convs with small output channels.

    // For immediate testing, let's use aclnnMatmul and see if we can pass
    // cubeMathType to handle the transpose internally...

    // Looking at aclnnMatmul more carefully:
    // aclnnMatmulGetWorkspaceSize(const aclTensor *mat1, const aclTensor *mat2,
    //                              aclTensor *out, int8_t cubeMathType,
    //                              uint64_t *workspaceSize, aclOpExecutor **executor)
    // mat1 @ mat2, no transpose support.

    // Let me try using the weight as [C_in, C_out] by swapping the shape dimensions
    // in the tensor descriptor. This is a "view" transpose which might not work
    // because the data is still stored in [C_out, C_in] order.

    // Actually wait - for a 1x1 conv, the weight is [C_out, C_in, 1, 1].
    // If we just view it as [C_out, C_in] (2D), that's correct.
    // For matmul we need: input [NHW, C_in] @ weight.T [C_in, C_out] = output [NHW, C_out]

    // The weight is stored as [C_out, C_in] in memory.
    // For matmul to work, we need the weight to be [C_in, C_out].

    // TEMPORARY FIX: Use aclnnMm with the wrong orientation and accept the bug
    // to see if the API works. We'll fix the transpose later.

    // NO - let's do it right.

    // The weight buffer contains data in [C_out, C_in] order.
    // We need to access it as if it were [C_in, C_out].
    // This requires actual data transposition, not just a view change.

    // Since we can't easily transpose at runtime, let's modify the constructor
    // to store weights in transposed form for small outChannels.

    // For NOW, let's try a different approach: use aclnnLinear
    // aclnnLinear does: output = input @ weight.T + bias
    // This is EXACTLY what we need!

    // But I don't have aclnnLinear header included. Let me check if it's available.
    // It might be in aclnn_linear.h or aclnnop/aclnn_linear.h

    // Since I can't easily add new headers right now, let's use a workaround:
    // Store the input/output as 2D and use the Conv2d API with groups=inChannels
    // which effectively does a 1x1 conv... no, that's wrong too.

    // OK, FINAL APPROACH for now:
    // Just use the convolution API but pad output channels to 16
    // Actually, that requires output buffer changes...

    // SIMPLEST: Let me just try aclnnMatmul with the weight as-is and see what error we get.
    // Maybe it will work with different cubeMathType?

    (void)workspaceBytes; (void)workspaceBuf; (void)accumulate;

    // Use aclnnMatmul: output = mat1 @ mat2
    // mat1 = input [NHW, C_in]
    // mat2 = weight [C_out, C_in] - WRONG, should be [C_in, C_out]
    // But let's try anyway to see if the API works

    // Actually, let's swap the order: weight.T @ input.T = (input @ weight).T
    // weight @ input.T = [C_out, C_in] @ [C_in, NHW] = [C_out, NHW]
    // Then transpose to get [NHW, C_out]
    // But we'd need to transpose input first...

    // This is getting complicated. Let me just add runtime transpose.

    // Create a transposed weight buffer in workspace
    // weight is [C_out, C_in], we need [C_in, C_out]
    // This is a transpose operation

    // For the ownership conv: weight is [1, 128], we need [128, 1]
    // Actually for 1 output channel, the transpose is trivial!

    // When C_out = 1:
    // weight [1, C_in] = row vector
    // weight.T [C_in, 1] = column vector
    // input @ weight.T = [NHW, C_in] @ [C_in, 1] = [NHW, 1] ✓

    // But our weight tensor is [1, C_in], not [C_in, 1].
    // We need to transpose. For C_out=1, we can use a special case.

    // For C_out=1: weight is [1, C_in]
    // input @ weight.T = input @ [C_in, 1] = [NHW, 1]
    // But matmul does: input @ weight = [NHW, C_in] @ [1, C_in] - INVALID (inner dims don't match)

    // So we MUST transpose the weight.

    // Workaround: Use aclnnMm with weight reinterpreted as [C_in, 1]
    // with strides adjusted to read the same data but in transposed order.
    // For a [1, C_in] tensor stored row-major, stride is [C_in, 1].
    // For a [C_in, 1] view of the same data, we need stride [1, 1] but that's wrong.

    // Actually, let me think about this differently.
    // Weight data: w[0][0], w[0][1], ..., w[0][C_in-1]  (one row)
    // We want to view as [C_in, 1] column:
    //   result[0][0] = w[0][0]
    //   result[1][0] = w[0][1]
    //   ...
    //   result[C_in-1][0] = w[0][C_in-1]

    // This IS a valid reinterpretation if we use the right strides!
    // Original stride for [1, C_in]: [C_in, 1]
    // New stride for [C_in, 1] view: [1, 1]  <- each row is 1 element, each col is 1 element stride

    // Wait no. For row-major [1, C_in]:
    //   element [i][j] is at offset i*C_in + j
    //   stride[0] = C_in, stride[1] = 1

    // For [C_in, 1] column view of same data:
    //   element [i][0] should be at offset i (reading sequentially)
    //   stride[0] = 1, stride[1] = 1

    // Yes! For a single-row tensor [1, C_in], viewing it as [C_in, 1] column
    // with stride [1, 1] would read elements sequentially, which is correct!

    // But wait, for [C_in, 1] with stride [1, 1], the storage size is still
    // computed based on shape. C_in * 1 * element_size. But we only have C_in elements,
    // so that's correct.

    // Let me create the weight tensor with transposed shape and adjusted strides.

    // For the general case where C_out might not be 1, we need a different approach.
    // Let me check if C_out is always 1 for the failing case...

    // From the error: outChannels=1. So for the immediate fix, let's handle C_out=1.

    // For C_out=1: Create weight tensor as [C_in, 1] with stride [1, 1]
    // This is a column vector view of the row vector data.

    // Actually, I realize aclCreateTensor allows specifying strides!
    // Let me use that to create a transposed view.

    // For weight data [C_out, C_in] stored row-major:
    // Shape: [C_out, C_in], Strides: [C_in, 1]

    // To view as [C_in, C_out] (transposed):
    // Shape: [C_in, C_out], Strides: [1, C_in]  <- swap the strides!

    // Let's try this approach.

    // Create transposed weight view
    vector<int64_t> weightTransShape = {C_in, C_out};
    vector<int64_t> weightTransStrides = {1, C_in};  // Swapped from [C_out, C_in]'s [C_in, 1]
    vector<int64_t> weightStorageShape = {C_out, C_in};  // Original physical layout

    aclTensor* weightTransTensor = aclCreateTensor(
      weightTransShape.data(),                      // viewDims
      static_cast<uint64_t>(weightTransShape.size()), // viewDimsNum
      dtype,                                        // dataType
      weightTransStrides.data(),                    // stride
      static_cast<int64_t>(0),                      // offset
      ACL_FORMAT_ND,                                // format
      weightStorageShape.data(),                    // storageDims (original physical layout)
      static_cast<uint64_t>(weightStorageShape.size()), // storageDimsNum
      filterBuf                                     // data
    );
    localTensors.push_back(weightTransTensor);

    // Now do matmul: input [NHW, C_in] @ weightTrans [C_in, C_out] = output [NHW, C_out]
    uint64_t matmulWsSize = 0;
    aclOpExecutor* matmulExecutor = nullptr;

    aclnnStatus status = aclnnMatmulGetWorkspaceSize(
      input2DTensor,
      weightTransTensor,
      output2DTensor,
      cubeMathType,
      &matmulWsSize,
      &matmulExecutor
    );

    if(status != ACLNN_SUCCESS) {
      for(aclTensor* t : localTensors) destroyAclTensor(t);
      throw StringError("aclnnMatmulGetWorkspaceSize failed for 1x1 conv " + name +
        " via MatMul path with error: " + to_string(status));
    }

    status = aclnnMatmul(workspaceBuf, matmulWsSize, matmulExecutor, stream);

    for(aclTensor* t : localTensors) destroyAclTensor(t);

    if(status != ACLNN_SUCCESS) {
      throw StringError("aclnnMatmul failed for 1x1 conv " + name + " with error: " + to_string(status));
    }
  }

  void applyViaConvolution(
    ComputeHandle* handle,
    aclrtStream stream,
    int batchSize,
    int nnXLen,
    int nnYLen,
    bool accumulate,
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    // Create tensors using cached descriptors (or fresh if handle is null for test functions)
    vector<int64_t> inputShape = {batchSize, inChannels, nnYLen, nnXLen};
    vector<int64_t> outputShape = {batchSize, outChannels, nnYLen, nnXLen};
    vector<int64_t> weightShape = {outChannels, inChannels, convYSize, convXSize};

    // Track locally-created tensors for cleanup when handle is nullptr
    vector<aclTensor*> localTensors;

    aclTensor* inputTensor;
    aclTensor* weightTensor;
    aclTensor* outputTensor;

    if(handle != nullptr) {
      inputTensor = handle->tensorCache.get(inputBuf, inputShape, dtype, ACL_FORMAT_NCHW);
      weightTensor = handle->tensorCache.get(filterBuf, weightShape, dtype, ACL_FORMAT_NCHW);
      outputTensor = handle->tensorCache.get(outputBuf, outputShape, dtype, ACL_FORMAT_NCHW);
    } else {
      inputTensor = createAclTensor(inputBuf, inputShape, dtype, ACL_FORMAT_NCHW);
      weightTensor = createAclTensor(filterBuf, weightShape, dtype, ACL_FORMAT_NCHW);
      outputTensor = createAclTensor(outputBuf, outputShape, dtype, ACL_FORMAT_NCHW);
      localTensors.push_back(inputTensor);
      localTensors.push_back(weightTensor);
      localTensors.push_back(outputTensor);
    }

    // Note: accumulate mode is not fully implemented yet
    (void)accumulate;

    // Phase 1: Get workspace size
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus status = aclnnConvolutionGetWorkspaceSize(
      inputTensor,
      weightTensor,
      nullptr,        // bias
      stridesArr,
      paddingsArr,
      dilationsArr,
      false,          // transposed
      outputPaddingArr,
      (int64_t)1,     // groups
      outputTensor,
      cubeMathType,      // cubeMathType: 0 for native FP16, 1 for FP32->FP16 conversion
      &wsSize,
      &executor
    );

    if(status != ACLNN_SUCCESS) {
      for(aclTensor* t : localTensors) destroyAclTensor(t);
      throw StringError("aclnnConvolutionGetWorkspaceSize failed for layer " + name +
        " with error: " + to_string(status) +
        " batchSize=" + to_string(batchSize) +
        " inChannels=" + to_string(inChannels) +
        " outChannels=" + to_string(outChannels) +
        " nnXLen=" + to_string(nnXLen) +
        " nnYLen=" + to_string(nnYLen) +
        " convYSize=" + to_string(convYSize) +
        " convXSize=" + to_string(convXSize) +
        " dilationY=" + to_string(dilationY) +
        " dilationX=" + to_string(dilationX) +
        " dtype=" + to_string((int)dtype) +
        " cubeMathType=" + to_string((int)cubeMathType));
    }

    // Phase 2: Execute
    status = aclnnConvolution(workspaceBuf, wsSize, executor, stream);

    // Cleanup local tensors if handle was nullptr
    for(aclTensor* t : localTensors) destroyAclTensor(t);

    if(status != ACLNN_SUCCESS) {
      throw StringError("aclnnConvolution failed for layer " + name + " with error: " + to_string(status));
    }

    (void)workspaceBytes;
  }
};

// BatchNormLayer - merged scale+bias with optional activation
struct BatchNormLayer {
  const string name;
  const int numChannels;
  const int activation;  // ACTIVATION_IDENTITY, RELU, or MISH

  void* mergedScaleBuf;  // Device memory
  void* mergedBiasBuf;   // Device memory
  bool useFP16;
  int nnXLen;
  int nnYLen;

  BatchNormLayer() = delete;
  BatchNormLayer(const BatchNormLayer&) = delete;
  BatchNormLayer& operator=(const BatchNormLayer&) = delete;

  BatchNormLayer(const BatchNormLayerDesc* desc, const ActivationLayerDesc* actDesc, int nnX, int nnY, bool useFP16_)
    : name(desc->name),
      numChannels(desc->numChannels),
      activation(actDesc ? actDesc->activation : ACTIVATION_IDENTITY),
      useFP16(useFP16_),
      nnXLen(nnX),
      nnYLen(nnY)
  {
    // Allocate and copy merged scale and bias with native FP16 conversion
    if(useFP16) {
      mergedScaleBuf = ascendMallocAndCopyFP16(desc->mergedScale);
      mergedBiasBuf = ascendMallocAndCopyFP16(desc->mergedBias);
    } else {
      size_t scaleBytes = desc->mergedScale.size() * sizeof(float);
      size_t biasBytes = desc->mergedBias.size() * sizeof(float);
      mergedScaleBuf = ascendMallocAndCopy(desc->mergedScale.data(), scaleBytes);
      mergedBiasBuf = ascendMallocAndCopy(desc->mergedBias.data(), biasBytes);
    }
  }

  ~BatchNormLayer() {
    ascendFree(mergedScaleBuf);
    ascendFree(mergedBiasBuf);
  }

  void apply(
    ComputeHandle* handle,
    aclrtStream stream,
    int batchSize,
    void* inputBuf,
    const void* maskBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    // BatchNorm: output = input * scale + bias, then apply activation
    // scale and bias are (numChannels,) shaped, need to broadcast to (N, C, H, W)

    aclDataType dtype = useFP16 ? ACL_FLOAT16 : ACL_FLOAT;

    // Step 1: Multiply input by scale (opIndex 0)
    // input: (N, C, H, W), scale: (1, C, 1, 1) -> output: (N, C, H, W)
    {
      vector<int64_t> inputShape = {batchSize, numChannels, nnYLen, nnXLen};
      vector<int64_t> scaleShape = {1, numChannels, 1, 1};  // Broadcast to NCHW

      aclTensor* inputTensor = handle->tensorCache.get(inputBuf, inputShape, dtype, ACL_FORMAT_NCHW);
      aclTensor* scaleTensor = handle->tensorCache.get(mergedScaleBuf, scaleShape, dtype, ACL_FORMAT_NCHW);
      aclTensor* outputTensor = handle->tensorCache.get(outputBuf, inputShape, dtype, ACL_FORMAT_NCHW);

      uint64_t mulWsSize = 0;
      aclOpExecutor* mulExecutor = nullptr;

      aclnnStatus status = aclnnMulGetWorkspaceSize(inputTensor, scaleTensor, outputTensor, &mulWsSize, &mulExecutor);
      if(status != ACLNN_SUCCESS) {
        throw StringError("aclnnMulGetWorkspaceSize failed for BatchNorm " + name + " with error: " + to_string(status));
      }

      status = aclnnMul(workspaceBuf, mulWsSize, mulExecutor, stream);
      if(status != ACLNN_SUCCESS) {
        throw StringError("aclnnMul failed for BatchNorm " + name + " with error: " + to_string(status));
      }
    }

    // Step 2: Add bias
    {
      vector<int64_t> outputShape = {batchSize, numChannels, nnYLen, nnXLen};
      vector<int64_t> biasShape = {1, numChannels, 1, 1};  // Broadcast to NCHW

      aclTensor* outputTensor = handle->tensorCache.get(outputBuf, outputShape, dtype, ACL_FORMAT_NCHW);
      aclTensor* biasTensor = handle->tensorCache.get(mergedBiasBuf, biasShape, dtype, ACL_FORMAT_NCHW);

      uint64_t addWsSize = 0;
      aclOpExecutor* addExecutor = nullptr;

      aclnnStatus status = aclnnAddGetWorkspaceSize(outputTensor, biasTensor, handle->alphaOneScalar, outputTensor, &addWsSize, &addExecutor);
      if(status != ACLNN_SUCCESS) {
        throw StringError("aclnnAddGetWorkspaceSize failed for BatchNorm " + name + " with error: " + to_string(status));
      }

      status = aclnnAdd(workspaceBuf, addWsSize, addExecutor, stream);
      if(status != ACLNN_SUCCESS) {
        throw StringError("aclnnAdd failed for BatchNorm " + name + " with error: " + to_string(status));
      }
    }

    // Step 3: Apply activation
    if(activation == ACTIVATION_RELU) {
      vector<int64_t> outputShape = {batchSize, numChannels, nnYLen, nnXLen};

      aclTensor* outputTensor = handle->tensorCache.get(outputBuf, outputShape, dtype, ACL_FORMAT_NCHW);

      uint64_t reluWsSize = 0;
      aclOpExecutor* reluExecutor = nullptr;

      aclnnStatus status = aclnnInplaceReluGetWorkspaceSize(outputTensor, &reluWsSize, &reluExecutor);
      if(status != ACLNN_SUCCESS) {
        throw StringError("aclnnInplaceReluGetWorkspaceSize failed for BatchNorm " + name + " with error: " + to_string(status));
      }

      status = aclnnInplaceRelu(workspaceBuf, reluWsSize, reluExecutor, stream);
      if(status != ACLNN_SUCCESS) {
        throw StringError("aclnnInplaceRelu failed for BatchNorm " + name + " with error: " + to_string(status));
      }
    }
    // MISH activation using native ACLNN operator
    else if(activation == ACTIVATION_MISH || activation == ACTIVATION_MISH_SCALE8) {
      vector<int64_t> outputShape = {batchSize, numChannels, nnYLen, nnXLen};

      aclTensor* outputTensor = handle->tensorCache.get(outputBuf, outputShape, dtype, ACL_FORMAT_NCHW);

      uint64_t mishWsSize = 0;
      aclOpExecutor* mishExecutor = nullptr;

      aclnnStatus status = aclnnInplaceMishGetWorkspaceSize(outputTensor, &mishWsSize, &mishExecutor);
      if(status != ACLNN_SUCCESS) {
        throw StringError("aclnnInplaceMishGetWorkspaceSize failed for BatchNorm " + name + " with error: " + to_string(status));
      }

      status = aclnnInplaceMish(workspaceBuf, mishWsSize, mishExecutor, stream);
      if(status != ACLNN_SUCCESS) {
        throw StringError("aclnnInplaceMish failed for BatchNorm " + name + " with error: " + to_string(status));
      }

      // For ACTIVATION_MISH_SCALE8, scale the output by 8.0
      if(activation == ACTIVATION_MISH_SCALE8) {
        aclScalar* scaleScalar = createFloatScalar(8.0f);

        uint64_t mulWsSize = 0;
        aclOpExecutor* mulExecutor = nullptr;

        status = aclnnInplaceMulsGetWorkspaceSize(outputTensor, scaleScalar, &mulWsSize, &mulExecutor);
        if(status != ACLNN_SUCCESS) {
          aclDestroyScalar(scaleScalar);
          throw StringError("aclnnInplaceMulsGetWorkspaceSize failed for MISH_SCALE8 " + name + " with error: " + to_string(status));
        }

        status = aclnnInplaceMuls(workspaceBuf, mulWsSize, mulExecutor, stream);
        aclDestroyScalar(scaleScalar);
        if(status != ACLNN_SUCCESS) {
          throw StringError("aclnnInplaceMuls failed for MISH_SCALE8 " + name + " with error: " + to_string(status));
        }
      }
    }

    // Step 4: Apply mask if provided
    if(maskBuf != nullptr) {
      vector<int64_t> outputShape = {batchSize, numChannels, nnYLen, nnXLen};
      vector<int64_t> maskShape = {batchSize, 1, nnYLen, nnXLen};

      aclTensor* outputTensor = handle->tensorCache.get(outputBuf, outputShape, dtype, ACL_FORMAT_NCHW);
      aclTensor* maskTensor = handle->tensorCache.get(const_cast<void*>(maskBuf), maskShape, dtype, ACL_FORMAT_NCHW);

      uint64_t maskWsSize = 0;
      aclOpExecutor* maskExecutor = nullptr;

      aclnnStatus status = aclnnInplaceMulGetWorkspaceSize(outputTensor, maskTensor, &maskWsSize, &maskExecutor);
      if(status != ACLNN_SUCCESS) {
        throw StringError("aclnnInplaceMulGetWorkspaceSize failed for BatchNorm " + name + " with error: " + to_string(status));
      }

      status = aclnnInplaceMul(workspaceBuf, maskWsSize, maskExecutor, stream);
      if(status != ACLNN_SUCCESS) {
        throw StringError("aclnnInplaceMul failed for BatchNorm " + name + " with error: " + to_string(status));
      }
    }

    (void)workspaceBytes;
  }
};

// MatMulLayer - matrix multiplication
struct MatMulLayer {
  const string name;
  const int inChannels;
  const int outChannels;

  void* matBuf;  // Device memory for weights
  bool useFP16;
  int8_t cubeMathType;  // 0=KEEP_DTYPE (native FP16), 1=ALLOW_FP32_DOWN_PRECISION

  MatMulLayer() = delete;
  MatMulLayer(const MatMulLayer&) = delete;
  MatMulLayer& operator=(const MatMulLayer&) = delete;

  MatMulLayer(const MatMulLayerDesc* desc, bool useFP16_)
    : name(desc->name),
      inChannels(desc->inChannels),
      outChannels(desc->outChannels),
      useFP16(useFP16_)
  {
    // Weights are in (inC, outC) format
    // Allocate and copy with native FP16 conversion for optimal performance
    if(useFP16) {
      matBuf = ascendMallocAndCopyFP16(desc->weights.data(), desc->weights.size());
      cubeMathType = 0;  // KEEP_DTYPE - weights are already native FP16
    } else {
      size_t weightBytes = desc->weights.size() * sizeof(float);
      matBuf = ascendMallocAndCopy(desc->weights.data(), weightBytes);
      cubeMathType = 1;  // ALLOW_FP32_DOWN_PRECISION - let CANN convert FP32->FP16
    }
  }

  ~MatMulLayer() {
    ascendFree(matBuf);
  }

  void apply(
    ComputeHandle* handle,
    aclrtStream stream,
    int batchSize,
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    // Matrix multiplication: output = input @ weights^T
    // input: (batch, inC), weights: (inC, outC), output: (batch, outC)
    // Note: KataGo stores weights in (inC, outC) format
    // For matmul with input (N, inC) @ weights (inC, outC) -> output (N, outC)

    aclDataType dtype = useFP16 ? ACL_FLOAT16 : ACL_FLOAT;

    // Create input tensor: (N, inC)
    vector<int64_t> inputShape = {batchSize, inChannels};
    aclTensor* inputTensor = handle->tensorCache.get(inputBuf, inputShape, dtype, ACL_FORMAT_ND);

    // Create weight tensor: (inC, outC)
    vector<int64_t> weightShape = {inChannels, outChannels};
    aclTensor* weightTensor = handle->tensorCache.get(matBuf, weightShape, dtype, ACL_FORMAT_ND);

    // Create output tensor: (N, outC)
    vector<int64_t> outputShape = {batchSize, outChannels};
    aclTensor* outputTensor = handle->tensorCache.get(outputBuf, outputShape, dtype, ACL_FORMAT_ND);

    // Phase 1: Get workspace size
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus status = aclnnMatmulGetWorkspaceSize(
      inputTensor,
      weightTensor,
      outputTensor,
      cubeMathType,       // cubeMathType: 0 for native FP16, 1 for FP32->FP16 conversion
      &wsSize,
      &executor
    );

    if(status != ACLNN_SUCCESS) {
      throw StringError("aclnnMatmulGetWorkspaceSize failed for MatMul " + name + " with error: " + to_string(status));
    }

    // Verify workspace is sufficient
    if(wsSize > workspaceBytes) {
      throw StringError("MatMul " + name + " requires more workspace: " + to_string(wsSize) + " > " + to_string(workspaceBytes));
    }

    // Phase 2: Execute
    status = aclnnMatmul(workspaceBuf, wsSize, executor, stream);

    if(status != ACLNN_SUCCESS) {
      throw StringError("aclnnMatmul failed for MatMul " + name + " with error: " + to_string(status));
    }
  }
};

// MatBiasLayer - bias addition
struct MatBiasLayer {
  const string name;
  const int numChannels;

  void* biasBuf;  // Device memory for bias
  bool useFP16;

  MatBiasLayer() = delete;
  MatBiasLayer(const MatBiasLayer&) = delete;
  MatBiasLayer& operator=(const MatBiasLayer&) = delete;

  MatBiasLayer(const MatBiasLayerDesc* desc, bool useFP16_)
    : name(desc->name),
      numChannels(desc->numChannels),
      useFP16(useFP16_)
  {
    // Allocate and copy bias with native FP16 conversion
    if(useFP16) {
      biasBuf = ascendMallocAndCopyFP16(desc->weights.data(), desc->weights.size());
    } else {
      size_t biasBytes = desc->weights.size() * sizeof(float);
      biasBuf = ascendMallocAndCopy(desc->weights.data(), biasBytes);
    }
  }

  ~MatBiasLayer() {
    ascendFree(biasBuf);
  }

  void apply(
    ComputeHandle* handle,
    aclrtStream stream,
    int batchSize,
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    // Add bias: output = input + bias
    // input: (batch, numChannels), bias: (numChannels,), output: (batch, numChannels)
    // Bias needs to be broadcast across the batch dimension

    aclDataType dtype = useFP16 ? ACL_FLOAT16 : ACL_FLOAT;

    // Create input tensor: (N, C)
    vector<int64_t> inputShape = {batchSize, numChannels};
    aclTensor* inputTensor = handle->tensorCache.get(inputBuf, inputShape, dtype, ACL_FORMAT_ND);

    // Create bias tensor: (1, C) - will broadcast across batch
    vector<int64_t> biasShape = {1, numChannels};
    aclTensor* biasTensor = handle->tensorCache.get(biasBuf, biasShape, dtype, ACL_FORMAT_ND);

    // Create output tensor: (N, C)
    vector<int64_t> outputShape = {batchSize, numChannels};
    aclTensor* outputTensor = handle->tensorCache.get(outputBuf, outputShape, dtype, ACL_FORMAT_ND);

    // Phase 1: Get workspace size
    uint64_t wsSize = 0;
    aclOpExecutor* executor = nullptr;

    aclnnStatus status = aclnnAddGetWorkspaceSize(inputTensor, biasTensor, handle->alphaOneScalar, outputTensor, &wsSize, &executor);
    if(status != ACLNN_SUCCESS) {
      throw StringError("aclnnAddGetWorkspaceSize failed for MatBias " + name + " with error: " + to_string(status));
    }

    // Verify workspace is sufficient
    if(wsSize > workspaceBytes) {
      throw StringError("MatBias " + name + " requires more workspace: " + to_string(wsSize) + " > " + to_string(workspaceBytes));
    }

    // Phase 2: Execute
    status = aclnnAdd(workspaceBuf, wsSize, executor, stream);

    if(status != ACLNN_SUCCESS) {
      throw StringError("aclnnAdd failed for MatBias " + name + " with error: " + to_string(status));
    }
  }
};

//---------------------------------------------------------------------------------
// Composite Layer Implementations
//---------------------------------------------------------------------------------

// NormActConv - BatchNorm + Activation + Conv fused pattern
struct NormActConv {
  const string name;
  unique_ptr<BatchNormLayer> bnLayer;
  unique_ptr<ConvLayer> convLayer;
  int numChannels;

  NormActConv() = delete;
  NormActConv(const NormActConv&) = delete;
  NormActConv& operator=(const NormActConv&) = delete;

  NormActConv(
    const BatchNormLayerDesc* bnDesc,
    const ActivationLayerDesc* actDesc,
    const ConvLayerDesc* convDesc,
    int nnXLen,
    int nnYLen,
    bool useFP16
  ) : name(bnDesc->name + "_" + convDesc->name), numChannels(convDesc->outChannels)
  {
    bnLayer = make_unique<BatchNormLayer>(bnDesc, actDesc, nnXLen, nnYLen, useFP16);
    convLayer = make_unique<ConvLayer>(convDesc, useFP16);
  }

  size_t requiredWorkspaceBytes(int batchSize, int nnXLen, int nnYLen, aclrtStream stream) const {
    return convLayer->requiredWorkspaceBytes(batchSize, nnXLen, nnYLen, stream);
  }

  void apply(
    ComputeHandle* handle,
    aclrtStream stream,
    int batchSize,
    int nnXLen,
    int nnYLen,
    const void* maskBuf,
    void* inputBuf,
    void* outputBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    // Apply BN + activation in-place on input
    bnLayer->apply(handle, stream, batchSize, inputBuf, maskBuf, inputBuf, workspaceBuf, workspaceBytes);

    // Then apply convolution
    convLayer->apply(handle, stream, batchSize, nnXLen, nnYLen, false, inputBuf, outputBuf, workspaceBuf, workspaceBytes);
  }
};

// ResidualBlock - Two NormActConvs with residual addition
struct ResidualBlock {
  const string name;
  unique_ptr<NormActConv> preNormActConv;
  unique_ptr<NormActConv> midNormActConv;
  unique_ptr<ConvLayer> finalConv;
  int numChannels;
  bool useFP16;

  ResidualBlock() = delete;
  ResidualBlock(const ResidualBlock&) = delete;
  ResidualBlock& operator=(const ResidualBlock&) = delete;

  ResidualBlock(
    const ResidualBlockDesc* desc,
    int nnXLen,
    int nnYLen,
    bool useFP16_
  ) : name(desc->name), numChannels(desc->finalConv.outChannels), useFP16(useFP16_)
  {
    preNormActConv = make_unique<NormActConv>(
      &desc->preBN, &desc->preActivation, &desc->regularConv, nnXLen, nnYLen, useFP16_);
    midNormActConv = make_unique<NormActConv>(
      &desc->midBN, &desc->midActivation, &desc->finalConv, nnXLen, nnYLen, useFP16_);
  }

  size_t requiredWorkspaceBytes(int batchSize, int nnXLen, int nnYLen, aclrtStream stream) const {
    size_t bytes = 0;
    bytes = max(bytes, preNormActConv->requiredWorkspaceBytes(batchSize, nnXLen, nnYLen, stream));
    bytes = max(bytes, midNormActConv->requiredWorkspaceBytes(batchSize, nnXLen, nnYLen, stream));
    return bytes;
  }

  void apply(
    ComputeHandle* handle,
    aclrtStream stream,
    int batchSize,
    int nnXLen,
    int nnYLen,
    const void* maskBuf,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* midBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    // Match CUDA's ResidualBlock pattern exactly:
    // Uses 3 separate buffers to avoid in-place convolution.
    //
    // CUDA flow (ResidualBlock):
    //   midIn = Conv(BN(trunkBuf))        [normActConv1]
    //   trunkBuf = Conv(BN(midIn)) + trunkBuf  [normActConv2 with accumulate]
    //
    // We avoid in-place operations by using trunkBuf, trunkScratchBuf, midBuf:
    //   1. preBN: trunkBuf → trunkScratchBuf (trunkBuf PRESERVED)
    //   2. preConv: trunkScratchBuf → midBuf
    //   3. midBN: midBuf → trunkScratchBuf (midBuf PRESERVED)
    //   4. midConv: trunkScratchBuf → midBuf
    //   5. Residual add: midBuf + trunkBuf → trunkBuf

    // Step 1: preBN + preActivation: trunkBuf → trunkScratchBuf
    preNormActConv->bnLayer->apply(handle, stream, batchSize, trunkBuf, maskBuf, trunkScratchBuf, workspaceBuf, workspaceBytes);

    // Step 2: preConv: trunkScratchBuf → midBuf
    preNormActConv->convLayer->apply(handle, stream, batchSize, nnXLen, nnYLen, false, trunkScratchBuf, midBuf, workspaceBuf, workspaceBytes);

    // Step 3: midBN + midActivation: midBuf → trunkScratchBuf
    midNormActConv->bnLayer->apply(handle, stream, batchSize, midBuf, maskBuf, trunkScratchBuf, workspaceBuf, workspaceBytes);

    // Step 4: midConv: trunkScratchBuf → midBuf
    midNormActConv->convLayer->apply(handle, stream, batchSize, nnXLen, nnYLen, false, trunkScratchBuf, midBuf, workspaceBuf, workspaceBytes);

    // Step 5: Residual add: midBuf + trunkBuf → trunkBuf
    aclDataType dtype = useFP16 ? ACL_FLOAT16 : ACL_FLOAT;

    vector<int64_t> addShape = {batchSize, numChannels, nnYLen, nnXLen};
    aclTensor* midTensor = handle->tensorCache.get(midBuf, addShape, dtype, ACL_FORMAT_NCHW);
    aclTensor* trunkTensor = handle->tensorCache.get(trunkBuf, addShape, dtype, ACL_FORMAT_NCHW);
    aclTensor* resultTensor = handle->tensorCache.get(trunkBuf, addShape, dtype, ACL_FORMAT_NCHW);

    uint64_t addWsSize = 0;
    aclOpExecutor* addExecutor = nullptr;

    aclnnStatus status = aclnnAddGetWorkspaceSize(midTensor, trunkTensor, handle->alphaOneScalar, resultTensor, &addWsSize, &addExecutor);
    if(status != ACLNN_SUCCESS) {
      throw StringError("aclnnAddGetWorkspaceSize failed for ResidualBlock " + name + " with error: " + to_string(status));
    }

    status = aclnnAdd(workspaceBuf, addWsSize, addExecutor, stream);

    if(status != ACLNN_SUCCESS) {
      throw StringError("aclnnAdd failed for ResidualBlock " + name + " with error: " + to_string(status));
    }

    (void)workspaceBytes;
  }
};

// GlobalPoolingResidualBlock - Residual block with global pooling branch
// Matches CUDA implementation pattern from cudabackend.cpp
struct GlobalPoolingResidualBlock {
  const string name;
  unique_ptr<BatchNormLayer> preBN;
  unique_ptr<ConvLayer> regularConv;
  unique_ptr<ConvLayer> gpoolConv;
  unique_ptr<BatchNormLayer> gpoolBN;
  unique_ptr<MatMulLayer> gpoolToBiasMul;
  unique_ptr<NormActConv> normActConv2;
  int numChannels;
  int regularChannels;
  int gpoolChannels;
  int nnXLen;
  int nnYLen;
  bool useFP16;

  GlobalPoolingResidualBlock() = delete;
  GlobalPoolingResidualBlock(const GlobalPoolingResidualBlock&) = delete;
  GlobalPoolingResidualBlock& operator=(const GlobalPoolingResidualBlock&) = delete;

  GlobalPoolingResidualBlock(
    const GlobalPoolingResidualBlockDesc* desc,
    int nnX,
    int nnY,
    bool useFP16_
  ) : name(desc->name),
      numChannels(desc->finalConv.outChannels),
      regularChannels(desc->regularConv.outChannels),
      gpoolChannels(desc->gpoolConv.outChannels),
      nnXLen(nnX),
      nnYLen(nnY),
      useFP16(useFP16_)
  {
    // preBN is applied to trunk (NOT part of NormActConv since regularConv is separate)
    preBN = make_unique<BatchNormLayer>(&desc->preBN, &desc->preActivation, nnX, nnY, useFP16_);
    regularConv = make_unique<ConvLayer>(&desc->regularConv, useFP16_);
    gpoolConv = make_unique<ConvLayer>(&desc->gpoolConv, useFP16_);
    gpoolBN = make_unique<BatchNormLayer>(&desc->gpoolBN, &desc->gpoolActivation, nnX, nnY, useFP16_);
    // CUDA: gpoolToBiasMul(cudaHandles,&desc->gpoolToBiasMul,false) - ALWAYS FP32
    gpoolToBiasMul = make_unique<MatMulLayer>(&desc->gpoolToBiasMul, false);
    normActConv2 = make_unique<NormActConv>(
      &desc->midBN, &desc->midActivation, &desc->finalConv, nnX, nnY, useFP16_);
  }

  size_t requiredWorkspaceBytes(int batchSize, aclrtStream stream) const {
    size_t bytes = 0;
    bytes = max(bytes, regularConv->requiredWorkspaceBytes(batchSize, nnXLen, nnYLen, stream));
    bytes = max(bytes, gpoolConv->requiredWorkspaceBytes(batchSize, nnXLen, nnYLen, stream));
    bytes = max(bytes, normActConv2->requiredWorkspaceBytes(batchSize, nnXLen, nnYLen, stream));
    return bytes;
  }

  void apply(
    ComputeHandle* handle,
    aclrtStream stream,
    int batchSize,
    const void* maskBuf,
    const float* maskSumBuf,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* midBuf,
    void* gpoolConvOutBuf,
    void* gpoolScratchBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    // Match CUDA's GlobalPoolingResidualBlock pattern exactly:
    // 1. preBN: trunkBuf -> trunkScratchBuf (preserves trunkBuf for residual)
    // 2. TWO parallel convs from trunkScratchBuf:
    //    a. regularConv: trunkScratchBuf -> midBuf (midBuf = regularOut)
    //    b. gpoolConv: trunkScratchBuf -> gpoolConvOutBuf (separate buffer)
    // 3. gpoolBN: gpoolConvOutBuf -> gpoolConvOutBuf (in-place)
    // 4. Global pooling: gpoolConvOutBuf -> gpoolScratchBuf intermediates (ALWAYS FP32)
    // 5. gpoolToBiasMul: gpoolConcat -> gpoolBias (FP32)
    // 6. Add gpoolBias to regularOut (midBuf) (broadcast)
    // 7. normActConv2 with residual: regularOut -> trunkBuf
    //
    // KEY: midBuf holds regularOut throughout. gpoolScratchBuf holds pooling intermediates.
    // workspaceBuf is ONLY used for ACLNN operator workspace (never for data storage).

    aclDataType dtype = useFP16 ? ACL_FLOAT16 : ACL_FLOAT;
    size_t poolElts = (size_t)batchSize * gpoolChannels;
    size_t poolBytesDtype = useFP16 ? (poolElts * sizeof(aclFloat16)) : (poolElts * sizeof(float));
    size_t poolBytesFP32 = poolElts * sizeof(float);

    // Layout in gpoolScratchBuf:
    //   [0..poolBytesDtype)                          = meanPool result
    //   [poolBytesDtype..+poolBytesFP32)              = meanFP32
    //   [+poolBytesFP32..+poolBytesDtype)             = maxPool result
    //   [+poolBytesDtype..+poolBytesFP32)             = maxFP32
    //   [+poolBytesFP32..+poolBytesFP32)              = scaledMean
    //   [+poolBytesFP32..+3*poolBytesFP32)            = gpoolConcat
    //   [+3*poolBytesFP32..+regularChannels*batch*4]  = gpoolBias
    void* meanPoolBuf = gpoolScratchBuf;
    void* meanFP32Buf = (char*)meanPoolBuf + poolBytesDtype;
    void* maxPoolBuf = (char*)meanFP32Buf + poolBytesFP32;
    void* maxFP32Buf = (char*)maxPoolBuf + poolBytesDtype;
    void* scaledMeanBuf = (char*)maxFP32Buf + poolBytesFP32;
    void* gpoolConcatBuf = (char*)scaledMeanBuf + poolBytesFP32;
    void* gpoolBiasBuf = (char*)gpoolConcatBuf + (size_t)gpoolChannels * 3 * batchSize * sizeof(float);

    // Step 1: preBN on trunk -> trunkScratchBuf (trunkBuf preserved for residual)
    preBN->apply(handle, stream, batchSize, trunkBuf, maskBuf, trunkScratchBuf, workspaceBuf, workspaceBytes);

    // Step 2a: regularConv: trunkScratchBuf -> midBuf (midBuf = regularOut)
    regularConv->apply(handle, stream, batchSize, nnXLen, nnYLen, false, trunkScratchBuf, midBuf, workspaceBuf, workspaceBytes);

    // Step 2b: gpoolConv: trunkScratchBuf -> gpoolConvOutBuf
    // Uses separate buffer to avoid in-place convolution (CANN may not support it)
    gpoolConv->apply(handle, stream, batchSize, nnXLen, nnYLen, false, trunkScratchBuf, gpoolConvOutBuf, workspaceBuf, workspaceBytes);

    // Step 3: gpoolBN: gpoolConvOutBuf -> gpoolConvOutBuf (in-place BN is safe)
    gpoolBN->apply(handle, stream, batchSize, gpoolConvOutBuf, maskBuf, gpoolConvOutBuf, workspaceBuf, workspaceBytes);

    // Step 4: Global pooling on gpoolConvOutBuf -> intermediates in gpoolScratchBuf (ALWAYS FP32)
    // Computes [mean, scaledMean, max] -> [batch, gpoolChannels * 3]
    {
      // 4a: Mean pooling
      aclTensor* gpoolOutTensorND = handle->tensorCache.get(gpoolConvOutBuf, {batchSize, gpoolChannels, nnYLen, nnXLen}, dtype, ACL_FORMAT_ND);
      aclTensor* meanPoolTensorND = handle->tensorCache.get(meanPoolBuf, {batchSize, gpoolChannels, 1, 1}, dtype, ACL_FORMAT_ND);

      aclIntArray* meanReduceDims = createAclIntArray({2, 3});
      uint64_t meanWsSize = 0;
      aclOpExecutor* meanExecutor = nullptr;
      aclnnStatus status = aclnnMeanGetWorkspaceSize(gpoolOutTensorND, meanReduceDims, true, dtype, meanPoolTensorND, &meanWsSize, &meanExecutor);
      if(status == ACLNN_SUCCESS && meanWsSize <= workspaceBytes) {
        status = aclnnMean(workspaceBuf, meanWsSize, meanExecutor, stream);
      }
      aclDestroyIntArray(meanReduceDims);
      if(status != ACLNN_SUCCESS) {
        throw StringError("aclnnMean failed for gpool block mean pooling: " + to_string(status));
      }

      // 4b: Cast mean to FP32 if needed
      aclTensor* meanFP32Tensor;
      if(useFP16) {
        aclTensor* srcTensor = handle->tensorCache.get(meanPoolBuf, {batchSize, gpoolChannels, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
        meanFP32Tensor = handle->tensorCache.get(meanFP32Buf, {batchSize, gpoolChannels, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
        uint64_t castWsSize = 0;
        aclOpExecutor* castExecutor = nullptr;
        status = aclnnCastGetWorkspaceSize(srcTensor, ACL_FLOAT, meanFP32Tensor, &castWsSize, &castExecutor);
        if(status == ACLNN_SUCCESS && castWsSize <= workspaceBytes) {
          status = aclnnCast(workspaceBuf, castWsSize, castExecutor, stream);
        }
        if(status != ACLNN_SUCCESS) {
          throw StringError("aclnnCast failed for gpool mean FP16->FP32: " + to_string(status));
        }
      } else {
        meanFP32Tensor = handle->tensorCache.get(meanPoolBuf, {batchSize, gpoolChannels, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
        meanFP32Buf = meanPoolBuf;
      }

      // 4c: Max pooling
      aclTensor* maxPoolTensor = handle->tensorCache.get(maxPoolBuf, {batchSize, gpoolChannels, 1, 1}, dtype, ACL_FORMAT_ND);
      aclIntArray* reduceDims = createAclIntArray({2, 3});
      uint64_t maxWsSize = 0;
      aclOpExecutor* maxExecutor = nullptr;
      status = aclnnAmaxGetWorkspaceSize(gpoolOutTensorND, reduceDims, true, maxPoolTensor, &maxWsSize, &maxExecutor);
      if(status == ACLNN_SUCCESS && maxWsSize <= workspaceBytes) {
        status = aclnnAmax(workspaceBuf, maxWsSize, maxExecutor, stream);
      }
      aclDestroyIntArray(reduceDims);
      if(status != ACLNN_SUCCESS) {
        throw StringError("aclnnAmax failed for gpool block max pooling: " + to_string(status));
      }

      // 4d: Cast max to FP32 if needed
      aclTensor* maxFP32Tensor;
      if(useFP16) {
        aclTensor* srcTensor = handle->tensorCache.get(maxPoolBuf, {batchSize, gpoolChannels, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
        maxFP32Tensor = handle->tensorCache.get(maxFP32Buf, {batchSize, gpoolChannels, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
        uint64_t castWsSize = 0;
        aclOpExecutor* castExecutor = nullptr;
        status = aclnnCastGetWorkspaceSize(srcTensor, ACL_FLOAT, maxFP32Tensor, &castWsSize, &castExecutor);
        if(status == ACLNN_SUCCESS && castWsSize <= workspaceBytes) {
          status = aclnnCast(workspaceBuf, castWsSize, castExecutor, stream);
        }
        if(status != ACLNN_SUCCESS) {
          throw StringError("aclnnCast failed for gpool max FP16->FP32: " + to_string(status));
        }
      } else {
        maxFP32Tensor = handle->tensorCache.get(maxPoolBuf, {batchSize, gpoolChannels, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
        maxFP32Buf = maxPoolBuf;
      }

      // 4e: scaledMean = mean * (sqrt(area) - 14) * 0.1
      float sqrtArea = sqrtf((float)(nnXLen * nnYLen));
      float scale = (sqrtArea - 14.0f) * 0.1f;
      aclTensor* scaledMeanTensor = handle->tensorCache.get(scaledMeanBuf, {batchSize, gpoolChannels, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
      {
        aclScalar* scaleScalar = aclCreateScalar(&scale, ACL_FLOAT);
        uint64_t mulsWsSize = 0;
        aclOpExecutor* mulsExecutor = nullptr;
        status = aclnnMulsGetWorkspaceSize(meanFP32Tensor, scaleScalar, scaledMeanTensor, &mulsWsSize, &mulsExecutor);
        if(status == ACLNN_SUCCESS && mulsWsSize <= workspaceBytes) {
          status = aclnnMuls(workspaceBuf, mulsWsSize, mulsExecutor, stream);
        }
        aclDestroyScalar(scaleScalar);
        if(status != ACLNN_SUCCESS) {
          throw StringError("aclnnMuls failed for gpool scaledMean: " + to_string(status));
        }
      }

      // 4f: Concatenate [mean, scaledMean, max] -> gpoolConcatBuf using D2D copies
      // (aclnnCat has compatibility issues with CANN 9.0; D2D memcpy is equivalent and more reliable)
      {
        size_t segmentBytes = poolElts * sizeof(float);
        aclError copyErr = aclrtMemcpyAsync(gpoolConcatBuf, segmentBytes, meanFP32Buf, segmentBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        if(copyErr != ACL_SUCCESS)
          throw StringError("aclrtMemcpyAsync failed for gpool block concat segment 0: " + to_string(copyErr));
        copyErr = aclrtMemcpyAsync((char*)gpoolConcatBuf + segmentBytes, segmentBytes, scaledMeanBuf, segmentBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        if(copyErr != ACL_SUCCESS)
          throw StringError("aclrtMemcpyAsync failed for gpool block concat segment 1: " + to_string(copyErr));
        copyErr = aclrtMemcpyAsync((char*)gpoolConcatBuf + 2 * segmentBytes, segmentBytes, maxFP32Buf, segmentBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
        if(copyErr != ACL_SUCCESS)
          throw StringError("aclrtMemcpyAsync failed for gpool block concat segment 2: " + to_string(copyErr));
      }
    }

    // Step 5: gpoolToBiasMul: gpoolConcat (FP32) -> gpoolBias (FP32, in gpoolScratchBuf)
    gpoolToBiasMul->apply(handle, stream, batchSize, gpoolConcatBuf, gpoolBiasBuf, workspaceBuf, workspaceBytes);

    // Step 6: Add gpoolBias to regularOut (midBuf) - broadcast across spatial dims
    {
      vector<int64_t> regularOutShape = {batchSize, regularChannels, nnYLen, nnXLen};
      vector<int64_t> biasShape = {batchSize, regularChannels, 1, 1};

      aclTensor* regularOutTensor = handle->tensorCache.get(midBuf, regularOutShape, dtype, ACL_FORMAT_NCHW);
      aclTensor* biasTensor = handle->tensorCache.get(gpoolBiasBuf, biasShape, ACL_FLOAT, ACL_FORMAT_NCHW);
      aclTensor* resultTensor = handle->tensorCache.get(midBuf, regularOutShape, dtype, ACL_FORMAT_NCHW);

      uint64_t addWsSize = 0;
      aclOpExecutor* addExecutor = nullptr;
      aclnnStatus addStatus = aclnnAddGetWorkspaceSize(regularOutTensor, biasTensor, handle->alphaOneScalar, resultTensor, &addWsSize, &addExecutor);
      if(addStatus == ACLNN_SUCCESS) {
        addStatus = aclnnAdd(workspaceBuf, addWsSize, addExecutor, stream);
      }
      if(addStatus != ACLNN_SUCCESS) {
        throw StringError("aclnnAdd failed for gpool block bias add: " + to_string(addStatus));
      }
    }

    // Step 7: normActConv2 with residual
    // BN: midBuf -> trunkScratchBuf, Conv: trunkScratchBuf -> midBuf
    normActConv2->bnLayer->apply(handle, stream, batchSize, midBuf, maskBuf, trunkScratchBuf, workspaceBuf, workspaceBytes);
    normActConv2->convLayer->apply(handle, stream, batchSize, nnXLen, nnYLen, false, trunkScratchBuf, midBuf, workspaceBuf, workspaceBytes);

    // Step 8: Residual add: midBuf + trunkBuf -> trunkBuf
    {
      vector<int64_t> addShape = {batchSize, numChannels, nnYLen, nnXLen};
      aclTensor* midOutTensor = handle->tensorCache.get(midBuf, addShape, dtype, ACL_FORMAT_NCHW);
      aclTensor* trunkTensor = handle->tensorCache.get(trunkBuf, addShape, dtype, ACL_FORMAT_NCHW);
      aclTensor* resultTensor = handle->tensorCache.get(trunkBuf, addShape, dtype, ACL_FORMAT_NCHW);

      uint64_t addWsSize = 0;
      aclOpExecutor* addExecutor = nullptr;
      aclnnStatus status = aclnnAddGetWorkspaceSize(midOutTensor, trunkTensor, handle->alphaOneScalar, resultTensor, &addWsSize, &addExecutor);
      if(status == ACLNN_SUCCESS) {
        status = aclnnAdd(workspaceBuf, addWsSize, addExecutor, stream);
      }
      if(status != ACLNN_SUCCESS) {
        throw StringError("aclnnAdd failed for GlobalPoolingResidualBlock " + name + " with error: " + to_string(status));
      }
    }

    (void)maskSumBuf;
    (void)workspaceBytes;
  }
};

//---------------------------------------------------------------------------------
// BlockStack - forward declaration (defined after NestedBottleneckResidualBlock)
//---------------------------------------------------------------------------------
struct BlockStack;

//---------------------------------------------------------------------------------
// NestedBottleneckResidualBlock - bottleneck block with nested BlockStack
//---------------------------------------------------------------------------------

struct NestedBottleneckResidualBlock {
  const string name;
  unique_ptr<NormActConv> normActConv1;
  unique_ptr<BlockStack> innerBlocks;
  unique_ptr<NormActConv> normActConv2;
  int numChannels;
  int nnXLen;
  int nnYLen;
  bool useFP16;

  NestedBottleneckResidualBlock() = delete;
  NestedBottleneckResidualBlock(const NestedBottleneckResidualBlock&) = delete;
  NestedBottleneckResidualBlock& operator=(const NestedBottleneckResidualBlock&) = delete;

  NestedBottleneckResidualBlock(
    const NestedBottleneckResidualBlockDesc* desc,
    int nnX,
    int nnY,
    bool fp16
  );

  ~NestedBottleneckResidualBlock();

  size_t requiredWorkspaceBytes(int batchSize, aclrtStream stream) const;

  void apply(
    ComputeHandle* handle,
    aclrtStream stream,
    int batchSize,
    const void* maskBuf,
    float* maskSumBuf,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* midBuf,
    void* gpoolConvOutBuf,
    void* gpoolScratchBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const;
};

//---------------------------------------------------------------------------------
// BlockStack - container for mixed block types (used by trunk and NestedBottleneck)
//---------------------------------------------------------------------------------

struct BlockStack {
  vector<pair<int, unique_ptr_void>> blocks;
  int numBlocks;
  int trunkNumChannels;
  int nnXLen;
  int nnYLen;
  bool useFP16;

  BlockStack() = delete;
  BlockStack(const BlockStack&) = delete;
  BlockStack& operator=(const BlockStack&) = delete;

  BlockStack(
    int nBlocks,
    int trunkChannels,
    const vector<pair<int, unique_ptr_void>>& descBlocks,
    int nnX,
    int nnY,
    bool fp16
  ) : numBlocks(nBlocks),
      trunkNumChannels(trunkChannels),
      nnXLen(nnX),
      nnYLen(nnY),
      useFP16(fp16)
  {
    assert(numBlocks == (int)descBlocks.size());
    for(int i = 0; i < numBlocks; i++) {
      if(descBlocks[i].first == ORDINARY_BLOCK_KIND) {
        const ResidualBlockDesc* blockDesc = static_cast<const ResidualBlockDesc*>(descBlocks[i].second.get());
        unique_ptr_void blockPtr = make_unique_void(
          new ResidualBlock(blockDesc, nnX, nnY, fp16)
        );
        blocks.emplace_back(ORDINARY_BLOCK_KIND, std::move(blockPtr));
      }
      else if(descBlocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
        const GlobalPoolingResidualBlockDesc* blockDesc = static_cast<const GlobalPoolingResidualBlockDesc*>(descBlocks[i].second.get());
        unique_ptr_void blockPtr = make_unique_void(
          new GlobalPoolingResidualBlock(blockDesc, nnX, nnY, fp16)
        );
        blocks.emplace_back(GLOBAL_POOLING_BLOCK_KIND, std::move(blockPtr));
      }
      else if(descBlocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
        const NestedBottleneckResidualBlockDesc* blockDesc = static_cast<const NestedBottleneckResidualBlockDesc*>(descBlocks[i].second.get());
        unique_ptr_void blockPtr = make_unique_void(
          new NestedBottleneckResidualBlock(blockDesc, nnX, nnY, fp16)
        );
        blocks.emplace_back(NESTED_BOTTLENECK_BLOCK_KIND, std::move(blockPtr));
      }
      else {
        throw StringError("Unknown block kind in BlockStack: " + to_string(descBlocks[i].first));
      }
    }
  }

  ~BlockStack() {}

  size_t requiredWorkspaceBytes(int batchSize, aclrtStream stream) const {
    size_t bytes = 0;
    for(int i = 0; i < (int)blocks.size(); i++) {
      size_t b = 0;
      if(blocks[i].first == ORDINARY_BLOCK_KIND) {
        ResidualBlock* block = static_cast<ResidualBlock*>(blocks[i].second.get());
        b = block->requiredWorkspaceBytes(batchSize, nnXLen, nnYLen, stream);
      }
      else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
        GlobalPoolingResidualBlock* block = static_cast<GlobalPoolingResidualBlock*>(blocks[i].second.get());
        b = block->requiredWorkspaceBytes(batchSize, stream);
      }
      else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
        NestedBottleneckResidualBlock* block = static_cast<NestedBottleneckResidualBlock*>(blocks[i].second.get());
        b = block->requiredWorkspaceBytes(batchSize, stream);
      }
      bytes = max(bytes, b);
    }
    return bytes;
  }

  void apply(
    ComputeHandle* handle,
    aclrtStream stream,
    int batchSize,
    const void* maskBuf,
    float* maskSumBuf,
    void* trunkBuf,
    void* trunkScratchBuf,
    void* midBuf,
    void* gpoolConvOutBuf,
    void* gpoolScratchBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const {
    for(int i = 0; i < (int)blocks.size(); i++) {
      if(blocks[i].first == ORDINARY_BLOCK_KIND) {
        ResidualBlock* block = static_cast<ResidualBlock*>(blocks[i].second.get());
        block->apply(handle, stream, batchSize, nnXLen, nnYLen, maskBuf,
          trunkBuf, trunkScratchBuf, midBuf,
          workspaceBuf, workspaceBytes);
      }
      else if(blocks[i].first == GLOBAL_POOLING_BLOCK_KIND) {
        GlobalPoolingResidualBlock* block = static_cast<GlobalPoolingResidualBlock*>(blocks[i].second.get());
        block->apply(handle, stream, batchSize, maskBuf, maskSumBuf,
          trunkBuf, trunkScratchBuf, midBuf,
          gpoolConvOutBuf, gpoolScratchBuf,
          workspaceBuf, workspaceBytes);
      }
      else if(blocks[i].first == NESTED_BOTTLENECK_BLOCK_KIND) {
        NestedBottleneckResidualBlock* block = static_cast<NestedBottleneckResidualBlock*>(blocks[i].second.get());
        block->apply(handle, stream, batchSize, maskBuf, maskSumBuf,
          trunkBuf, trunkScratchBuf, midBuf,
          gpoolConvOutBuf, gpoolScratchBuf,
          workspaceBuf, workspaceBytes);
      }
    }
  }
};

//---------------------------------------------------------------------------------
// NestedBottleneckResidualBlock implementation (after BlockStack is defined)
//---------------------------------------------------------------------------------

NestedBottleneckResidualBlock::NestedBottleneckResidualBlock(
  const NestedBottleneckResidualBlockDesc* desc,
  int nnX,
  int nnY,
  bool fp16
) : name(desc->name),
    numChannels(desc->postConv.outChannels),
    nnXLen(nnX),
    nnYLen(nnY),
    useFP16(fp16)
{
  normActConv1 = make_unique<NormActConv>(
    &desc->preBN, &desc->preActivation, &desc->preConv, nnX, nnY, fp16);
  innerBlocks = make_unique<BlockStack>(
    desc->numBlocks, desc->preConv.outChannels, desc->blocks, nnX, nnY, fp16);
  normActConv2 = make_unique<NormActConv>(
    &desc->postBN, &desc->postActivation, &desc->postConv, nnX, nnY, fp16);
}

NestedBottleneckResidualBlock::~NestedBottleneckResidualBlock() {}

size_t NestedBottleneckResidualBlock::requiredWorkspaceBytes(int batchSize, aclrtStream stream) const {
  size_t bytes = 0;
  bytes = max(bytes, normActConv1->requiredWorkspaceBytes(batchSize, nnXLen, nnYLen, stream));
  bytes = max(bytes, innerBlocks->requiredWorkspaceBytes(batchSize, stream));
  bytes = max(bytes, normActConv2->requiredWorkspaceBytes(batchSize, nnXLen, nnYLen, stream));
  return bytes;
}

void NestedBottleneckResidualBlock::apply(
  ComputeHandle* handle,
  aclrtStream stream,
  int batchSize,
  const void* maskBuf,
  float* maskSumBuf,
  void* trunkBuf,
  void* trunkScratchBuf,
  void* midBuf,
  void* gpoolConvOutBuf,
  void* gpoolScratchBuf,
  void* workspaceBuf,
  size_t workspaceBytes
) const {
  // Match CUDA's NestedBottleneckResidualBlock::apply exactly:
  // 1. normActConv1: trunkBuf -> midBuf (BN+Act+Conv)
  // 2. innerBlocks: midBuf -> midBuf (recursive)
  // 3. normActConv2: midBuf -> trunkBuf with accumulate=true (residual add)

  // Step 1: normActConv1: trunkBuf -> midBuf
  // CUDA: normActConv1.apply(cudaHandles,batchSize,false,trunkBuf,trunkScratchBuf,mid.buf,maskBuf,...)
  normActConv1->bnLayer->apply(handle, stream, batchSize, trunkBuf, maskBuf, trunkScratchBuf, workspaceBuf, workspaceBytes);
  normActConv1->convLayer->apply(handle, stream, batchSize, nnXLen, nnYLen, false, trunkScratchBuf, midBuf, workspaceBuf, workspaceBytes);

  // Step 2: innerBlocks operates on midBuf (as trunk) with trunkScratchBuf as scratch
  // CUDA: blocks.apply(cudaHandles,scratch,batchSize,maskBuf,maskSumBuf,mid.buf,midScratch.buf,...)
  innerBlocks->apply(
    handle, stream, batchSize, maskBuf, maskSumBuf,
    midBuf, trunkScratchBuf, midBuf,
    gpoolConvOutBuf, gpoolScratchBuf,
    workspaceBuf, workspaceBytes
  );

  // Step 3: normActConv2 with residual add (accumulate=true)
  // CUDA: normActConv2.apply(cudaHandles,batchSize,true,mid.buf,midScratch.buf,trunkBuf,maskBuf,...)
  normActConv2->bnLayer->apply(handle, stream, batchSize, midBuf, maskBuf, trunkScratchBuf, workspaceBuf, workspaceBytes);
  normActConv2->convLayer->apply(handle, stream, batchSize, nnXLen, nnYLen, false, trunkScratchBuf, midBuf, workspaceBuf, workspaceBytes);

  // Residual add: midBuf + trunkBuf -> trunkBuf
  aclDataType dtype = useFP16 ? ACL_FLOAT16 : ACL_FLOAT;
  vector<int64_t> addShape = {batchSize, numChannels, nnYLen, nnXLen};
  aclTensor* midTensor = handle->tensorCache.get(midBuf, addShape, dtype, ACL_FORMAT_NCHW);
  aclTensor* trunkTensor = handle->tensorCache.get(trunkBuf, addShape, dtype, ACL_FORMAT_NCHW);
  aclTensor* resultTensor = handle->tensorCache.get(trunkBuf, addShape, dtype, ACL_FORMAT_NCHW);

  uint64_t addWsSize = 0;
  aclOpExecutor* addExecutor = nullptr;
  aclnnStatus status = aclnnAddGetWorkspaceSize(midTensor, trunkTensor, handle->alphaOneScalar, resultTensor, &addWsSize, &addExecutor);
  if(status != ACLNN_SUCCESS) {
    throw StringError("aclnnAddGetWorkspaceSize failed for NestedBottleneckResidualBlock " + name + " with error: " + to_string(status));
  }
  status = aclnnAdd(workspaceBuf, addWsSize, addExecutor, stream);
  if(status != ACLNN_SUCCESS) {
    throw StringError("aclnnAdd failed for NestedBottleneckResidualBlock " + name + " with error: " + to_string(status));
  }
}

//---------------------------------------------------------------------------------
// Model Structure
//---------------------------------------------------------------------------------

// Forward declarations for head structures
struct Trunk;
struct PolicyHead;
struct ValueHead;

struct Model {
  int numInputChannels;
  int numInputGlobalChannels;
  int numInputMetaChannels;
  int numPolicyChannels;
  int numValueChannels;
  int numScoreValueChannels;
  int numOwnershipChannels;
  int modelVersion;
  int trunkNumChannels;
  int nnXLen;
  int nnYLen;
  bool useFP16;

  // Trunk layers
  unique_ptr<ConvLayer> initialConv;
  unique_ptr<MatMulLayer> initialMatMul;
  unique_ptr<BatchNormLayer> trunkTipBN;
  vector<unique_ptr<ResidualBlock>> residualBlocks;
  vector<unique_ptr<GlobalPoolingResidualBlock>> gpoolBlocks;
  vector<unique_ptr<NestedBottleneckResidualBlock>> nestedBlocks;

  // Ordered block kinds - matches CUDA BlockStack iteration order
  vector<int> trunkBlockKinds;

  // Policy head layers
  unique_ptr<ConvLayer> p1Conv;
  unique_ptr<ConvLayer> g1Conv;
  unique_ptr<BatchNormLayer> g1BN;
  unique_ptr<MatMulLayer> gpoolToBiasMul;
  unique_ptr<BatchNormLayer> p1BN;
  unique_ptr<ConvLayer> p2Conv;
  unique_ptr<MatMulLayer> gpoolToPassMul;
  unique_ptr<MatBiasLayer> gpoolToPassBias;
  unique_ptr<MatMulLayer> gpoolToPassMul2;

  // Value head layers
  unique_ptr<ConvLayer> v1Conv;
  unique_ptr<BatchNormLayer> v1BN;
  unique_ptr<MatMulLayer> v2Mul;
  unique_ptr<MatBiasLayer> v2Bias;
  unique_ptr<MatMulLayer> v3Mul;
  unique_ptr<MatBiasLayer> v3Bias;
  unique_ptr<MatMulLayer> sv3Mul;
  unique_ptr<MatBiasLayer> sv3Bias;
  unique_ptr<ConvLayer> vOwnershipConv;

  // SGF Metadata encoder (optional)
  bool hasMetadataEncoder;
  unique_ptr<MatMulLayer> metaMul1;
  unique_ptr<MatBiasLayer> metaBias1;
  unique_ptr<MatMulLayer> metaMul2;
  unique_ptr<MatBiasLayer> metaBias2;
  unique_ptr<MatMulLayer> metaMul3;

  Model(const ModelDesc& desc, int nnX, int nnY, bool fp16);
  ~Model() {}

  void apply(
    ComputeHandle* handle,
    aclrtStream stream,
    int batchSize,
    bool requireExactNNLen,
    void* inputBuf,
    void* inputGlobalBuf,
    void* inputMetaBuf,
    float* policyPassBuf,
    float* policyBuf,
    float* valueBuf,
    float* scoreValueBuf,
    void* ownershipBuf,
    Buffers* buffers
  ) const;

  size_t requiredWorkspaceBytes(int maxBatchSize) const;

private:
  void applyTrunk(
    ComputeHandle* handle,
    aclrtStream stream,
    int batchSize,
    void* inputBuf,
    void* inputGlobalBuf,
    void* inputMetaBuf,
    void* trunkOutputBuf,
    void* maskBuf,
    float* maskSumBuf,
    void* scratchBuf,
    void* residualMidBuf,
    void* gpoolConvOutBuf,
    void* gpoolScratchBuf,
    void* workspaceBuf,
    size_t workspaceBytes
  ) const;

  void applyPolicyHead(
    ComputeHandle* handle,
    aclrtStream stream,
    int batchSize,
    const void* trunkOutputBuf,
    const void* maskBuf,
    const void* maskFloatBuf,
    float* maskSumBuf,
    float* policyPassBuf,
    float* policyBuf,
    void* scratchBuf,
    void* workspaceBuf,
    size_t workspaceBytes,
    Buffers* buffers
  ) const;

  void applyValueHead(
    ComputeHandle* handle,
    aclrtStream stream,
    int batchSize,
    const void* trunkOutputBuf,
    const void* maskBuf,
    float* maskSumBuf,
    float* valueBuf,
    float* scoreValueBuf,
    void* ownershipBuf,
    void* scratchBuf,
    void* workspaceBuf,
    size_t workspaceBytes,
    Buffers* buffers
  ) const;
};

Model::Model(const ModelDesc& desc, int nnX, int nnY, bool fp16)
  : numInputChannels(desc.numInputChannels),
    numInputGlobalChannels(desc.numInputGlobalChannels),
    numInputMetaChannels(desc.numInputMetaChannels),
    numPolicyChannels(desc.numPolicyChannels),
    numValueChannels(desc.numValueChannels),
    numScoreValueChannels(desc.numScoreValueChannels),
    numOwnershipChannels(desc.numOwnershipChannels),
    modelVersion(desc.modelVersion),
    trunkNumChannels(desc.trunk.trunkNumChannels),
    nnXLen(nnX),
    nnYLen(nnY),
    useFP16(fp16),
    hasMetadataEncoder(desc.numInputMetaChannels > 0)
{
  // Create trunk layers
  initialConv = make_unique<ConvLayer>(&desc.trunk.initialConv, fp16);
  initialMatMul = make_unique<MatMulLayer>(&desc.trunk.initialMatMul, fp16);

  // Create residual blocks in order
  for(const auto& blockPair : desc.trunk.blocks) {
    int blockKind = blockPair.first;
    trunkBlockKinds.push_back(blockKind);
    if(blockKind == ORDINARY_BLOCK_KIND) {
      const ResidualBlockDesc* rdesc = static_cast<const ResidualBlockDesc*>(blockPair.second.get());
      residualBlocks.push_back(make_unique<ResidualBlock>(rdesc, nnX, nnY, fp16));
    } else if(blockKind == GLOBAL_POOLING_BLOCK_KIND) {
      const GlobalPoolingResidualBlockDesc* gdesc = static_cast<const GlobalPoolingResidualBlockDesc*>(blockPair.second.get());
      gpoolBlocks.push_back(make_unique<GlobalPoolingResidualBlock>(gdesc, nnX, nnY, fp16));
    } else if(blockKind == NESTED_BOTTLENECK_BLOCK_KIND) {
      const NestedBottleneckResidualBlockDesc* ndesc = static_cast<const NestedBottleneckResidualBlockDesc*>(blockPair.second.get());
      nestedBlocks.push_back(make_unique<NestedBottleneckResidualBlock>(ndesc, nnX, nnY, fp16));
    }
  }

  // Create trunk tip BN
  trunkTipBN = make_unique<BatchNormLayer>(&desc.trunk.trunkTipBN, &desc.trunk.trunkTipActivation, nnX, nnY, fp16);

  // Create policy head layers
  // CRITICAL: Match CUDA backend behavior exactly:
  // - p1Conv and g1Conv use FP16 (they operate on trunk features)
  // - g1BN uses FP16 (operates on g1Conv output)
  // - gpoolToBiasMul, p1BN, p2Conv, gpoolToPassMul, gpoolToPassBias, gpoolToPassMul2
  //   ALL use FP32 (useFP16=false) to match CUDA's numerical precision
  // CUDA: gpoolToBiasMul(cudaHandles,&desc->gpoolToBiasMul,false)
  // CUDA: p1BN(cudaHandles,...,fp16=false)
  // CUDA: p2Conv(cudaHandles,&desc->p2Conv,fp16=false)
  p1Conv = make_unique<ConvLayer>(&desc.policyHead.p1Conv, fp16);
  g1Conv = make_unique<ConvLayer>(&desc.policyHead.g1Conv, fp16);
  g1BN = make_unique<BatchNormLayer>(&desc.policyHead.g1BN, &desc.policyHead.g1Activation, nnX, nnY, fp16);
  gpoolToBiasMul = make_unique<MatMulLayer>(&desc.policyHead.gpoolToBiasMul, false);  // ALWAYS FP32
  p1BN = make_unique<BatchNormLayer>(&desc.policyHead.p1BN, &desc.policyHead.p1Activation, nnX, nnY, false);  // ALWAYS FP32
  p2Conv = make_unique<ConvLayer>(&desc.policyHead.p2Conv, false);  // ALWAYS FP32
  gpoolToPassMul = make_unique<MatMulLayer>(&desc.policyHead.gpoolToPassMul, false);  // ALWAYS FP32
  gpoolToPassBias = make_unique<MatBiasLayer>(&desc.policyHead.gpoolToPassBias, false);  // ALWAYS FP32
  gpoolToPassMul2 = make_unique<MatMulLayer>(&desc.policyHead.gpoolToPassMul2, false);  // ALWAYS FP32

  // Create value head layers
  // CRITICAL: Match CUDA backend behavior exactly:
  // - v1Conv and v1BN use FP16 (they operate on trunk features)
  // - v2Mul, v2Bias, v3Mul, v3Bias, sv3Mul, sv3Bias ALL use FP32
  // CUDA: v2Mul(cudaHandles,&desc->v2Mul,false)
  // CUDA: v2Bias(cudaHandles,...,fp16=false)
  v1Conv = make_unique<ConvLayer>(&desc.valueHead.v1Conv, fp16);
  v1BN = make_unique<BatchNormLayer>(&desc.valueHead.v1BN, &desc.valueHead.v1Activation, nnX, nnY, fp16);
  v2Mul = make_unique<MatMulLayer>(&desc.valueHead.v2Mul, false);  // ALWAYS FP32
  v2Bias = make_unique<MatBiasLayer>(&desc.valueHead.v2Bias, false);  // ALWAYS FP32
  v3Mul = make_unique<MatMulLayer>(&desc.valueHead.v3Mul, false);  // ALWAYS FP32
  v3Bias = make_unique<MatBiasLayer>(&desc.valueHead.v3Bias, false);  // ALWAYS FP32
  sv3Mul = make_unique<MatMulLayer>(&desc.valueHead.sv3Mul, false);  // ALWAYS FP32
  sv3Bias = make_unique<MatBiasLayer>(&desc.valueHead.sv3Bias, false);  // ALWAYS FP32
  vOwnershipConv = make_unique<ConvLayer>(&desc.valueHead.vOwnershipConv, fp16);

  // Create metadata encoder layers if present
  if(hasMetadataEncoder && desc.trunk.sgfMetadataEncoder.metaEncoderVersion > 0) {
    const auto& meta = desc.trunk.sgfMetadataEncoder;
    metaMul1 = make_unique<MatMulLayer>(&meta.mul1, fp16);
    metaBias1 = make_unique<MatBiasLayer>(&meta.bias1, fp16);
    metaMul2 = make_unique<MatMulLayer>(&meta.mul2, fp16);
    metaBias2 = make_unique<MatBiasLayer>(&meta.bias2, fp16);
    metaMul3 = make_unique<MatMulLayer>(&meta.mul3, fp16);
  }
}

size_t Model::requiredWorkspaceBytes(int maxBatchSize) const {
  // Calculate maximum workspace needed across all operations
  size_t maxBytes = 0;

  // Initial conv workspace
  maxBytes = max(maxBytes, initialConv->requiredWorkspaceBytes(maxBatchSize, nnXLen, nnYLen, nullptr));

  // Residual blocks
  for(const auto& block : residualBlocks) {
    maxBytes = max(maxBytes, block->requiredWorkspaceBytes(maxBatchSize, nnXLen, nnYLen, nullptr));
  }

  // Nested bottleneck blocks
  for(const auto& block : nestedBlocks) {
    maxBytes = max(maxBytes, block->requiredWorkspaceBytes(maxBatchSize, nullptr));
  }

  // Policy head
  maxBytes = max(maxBytes, p1Conv->requiredWorkspaceBytes(maxBatchSize, nnXLen, nnYLen, nullptr));
  maxBytes = max(maxBytes, p2Conv->requiredWorkspaceBytes(maxBatchSize, nnXLen, nnYLen, nullptr));

  // Value head
  maxBytes = max(maxBytes, v1Conv->requiredWorkspaceBytes(maxBatchSize, nnXLen, nnYLen, nullptr));
  maxBytes = max(maxBytes, vOwnershipConv->requiredWorkspaceBytes(maxBatchSize, nnXLen, nnYLen, nullptr));

  // Add extra for intermediate tensors
  maxBytes += maxBatchSize * trunkNumChannels * nnXLen * nnYLen * sizeof(float) * 4;

  return maxBytes;
}

void Model::apply(
  ComputeHandle* handle,
  aclrtStream stream,
  int batchSize,
  bool requireExactNNLen,
  void* inputBuf,
  void* inputGlobalBuf,
  void* inputMetaBuf,
  float* policyPassBuf,
  float* policyBuf,
  float* valueBuf,
  float* scoreValueBuf,
  void* ownershipBuf,
  Buffers* buffers
) const {
  // Use pre-allocated intermediate buffers from Buffers struct
  void* trunkOutputBuf = buffers->trunkBuf;
  void* maskBuf = buffers->maskBuf;
  float* maskSumBuf = (float*)buffers->maskSumBuf;
  float* maskFloatBuf = (float*)buffers->maskFloatBuf;
  void* scratchBuf = buffers->scratchBuf;
  void* workspaceBuf = buffers->workspaceBuf;
  size_t workspaceBytes = buffers->workspaceBytes;

  // ================================================================
  // Step 1: Extract mask from input channel 0 and compute maskSum
  // ================================================================
  // The mask is stored in channel 0 of the input tensor.
  // In NCHW layout: input[n, 0, h, w] = mask value (1.0 = valid, 0.0 = padding)
  // maskSumBuf[n] = number of valid board positions for batch element n
  {
    aclDataType inputDtype = useFP16 ? ACL_FLOAT16 : ACL_FLOAT;
    // Create input tensor view restricted to channel 0
    // We can use aclCreateTensor with offset to select channel 0
    // For NCHW with shape [N, C, H, W], channel 0 starts at offset 0 with stride [C*H*W, H*W, W, 1]
    // We want shape [N, 1, H, W] with the same data pointer and stride [C*H*W, H*W, W, 1]
    int64_t cStride = nnXLen * nnYLen;
    int64_t inputStride[4] = {cStride * numInputChannels, cStride, (int64_t)nnXLen, 1};
    vector<int64_t> maskViewShape = {batchSize, 1, nnYLen, nnXLen};

    aclTensor* inputChannel0Tensor = aclCreateTensor(
      maskViewShape.data(),                     // viewDims: [N, 1, H, W]
      static_cast<uint64_t>(maskViewShape.size()), // viewDimsNum
      inputDtype,                               // dataType
      inputStride,                              // strides (same as full input)
      static_cast<int64_t>(0),                  // storageOffset
      ACL_FORMAT_NCHW,                          // format
      maskViewShape.data(),                     // storageDims
      static_cast<uint64_t>(maskViewShape.size()), // storageDimsNum
      inputBuf                                  // data (same as full input)
    );

    // Copy channel 0 to maskBuf (same dtype as input)
    size_t maskBytes = (size_t)batchSize * nnYLen * nnXLen * (useFP16 ? sizeof(aclFloat16) : sizeof(float));

    // Use aclrtMemcpy for device-to-device copy (aclnnCopy doesn't exist in CANN)
    // maskBuf is at offset 0 of inputBuf, so just copy from inputBuf to maskBuf
    aclError copyErr = aclrtMemcpyAsync(maskBuf, maskBytes, inputBuf, maskBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
    aclDestroyTensor(inputChannel0Tensor);
    if(copyErr != ACL_SUCCESS) {
      throw StringError("aclrtMemcpyAsync failed for mask extraction: " + to_string(copyErr));
    }

    // If requireExactNNLen, set maskBuf to NULL to skip masking in BN layers
    // (global pooling still needs maskSumBuf for normalization)
    if(requireExactNNLen) {
      maskBuf = nullptr;
    }

    // Compute maskSum: sum mask values across spatial dims for each batch element
    // maskFloatBuf is always float
    aclnnStatus status;
    if(useFP16) {
      // Cast mask from FP16 to FP32 using host-side conversion (aclnnCast is broken)
      size_t numMaskElts = (size_t)batchSize * nnYLen * nnXLen;
      castDeviceFP16ToFP32(stream, maskFloatBuf, maskBuf, numMaskElts);
    } else {
      // In FP32 mode, maskBuf and maskFloatBuf share the same data
      maskFloatBuf = (float*)maskBuf;
    }

    // Sum across spatial dimensions: [N, 1, H, W] -> [N, 1]
    // Use aclnnMean with multiplication by area to get sum
    // Or use aclnnSum with reduction
    aclTensor* maskSumInputTensor = createAclTensor(
      maskFloatBuf, {batchSize, 1, nnYLen, nnXLen}, ACL_FLOAT, ACL_FORMAT_ND);
    aclTensor* maskSumOutputTensor = createAclTensor(
      maskSumBuf, {batchSize, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    aclIntArray* sumReduceDims = createAclIntArray({2, 3});
    bool sumKeepDim = true;

    uint64_t sumWsSize = 0;
    aclOpExecutor* sumExecutor = nullptr;
    // Use aclnnMean then multiply by area to get sum (since aclnnSum may not exist)
    // Actually, let's use aclnnMean and multiply by HW
    aclTensor* meanOutput = createAclTensor(
      maskSumBuf, {batchSize, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    status = aclnnMeanGetWorkspaceSize(maskSumInputTensor, sumReduceDims, sumKeepDim, ACL_FLOAT, meanOutput, &sumWsSize, &sumExecutor);
    if(status == ACLNN_SUCCESS) {
      status = aclnnMean(workspaceBuf, sumWsSize, sumExecutor, stream);
    }
    aclDestroyIntArray(sumReduceDims);
    aclDestroyTensor(maskSumInputTensor);
    aclDestroyTensor(maskSumOutputTensor);
    if(status != ACLNN_SUCCESS) {
      throw StringError("aclnnMean failed for mask sum: " + to_string(status));
    }

    // maskSum = mean * H * W
    float area = (float)(nnXLen * nnYLen);
    aclScalar* areaScalar = createFloatScalar(area);
    aclTensor* maskSumTensor = createAclTensor(maskSumBuf, {batchSize, 1, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    uint64_t mulsWsSize = 0;
    aclOpExecutor* mulsExecutor = nullptr;
    status = aclnnInplaceMulsGetWorkspaceSize(maskSumTensor, areaScalar, &mulsWsSize, &mulsExecutor);
    if(status == ACLNN_SUCCESS) {
      status = aclnnInplaceMuls(workspaceBuf, mulsWsSize, mulsExecutor, stream);
    }
    aclDestroyScalar(areaScalar);
    aclDestroyTensor(maskSumTensor);
    aclDestroyTensor(meanOutput);
    if(status != ACLNN_SUCCESS) {
      throw StringError("aclnnInplaceMuls failed for mask sum scaling: " + to_string(status));
    }
  }

  // ================================================================
  // Step 2: Apply trunk
  // ================================================================
  applyTrunk(handle, stream, batchSize, inputBuf, inputGlobalBuf, inputMetaBuf, trunkOutputBuf, maskBuf, maskSumBuf, scratchBuf, buffers->residualMidBuf, buffers->gpoolConvOutBuf, buffers->gpoolScratchBuf, workspaceBuf, workspaceBytes);

  // ================================================================
  // Step 3: Apply policy head
  // ================================================================
  applyPolicyHead(handle, stream, batchSize, trunkOutputBuf, maskBuf, buffers->maskFloatBuf, maskSumBuf, policyPassBuf, policyBuf, scratchBuf, workspaceBuf, workspaceBytes, buffers);

  // ================================================================
  // Step 4: Apply value head
  // ================================================================
  applyValueHead(handle, stream, batchSize, trunkOutputBuf, maskBuf, maskSumBuf, valueBuf, scoreValueBuf, ownershipBuf, scratchBuf, workspaceBuf, workspaceBytes, buffers);
}

void Model::applyTrunk(
  ComputeHandle* handle,
  aclrtStream stream,
  int batchSize,
  void* inputBuf,
  void* inputGlobalBuf,
  void* inputMetaBuf,
  void* trunkOutputBuf,
  void* maskBuf,
  float* maskSumBuf,
  void* scratchBuf,
  void* residualMidBuf,
  void* gpoolConvOutBuf,
  void* gpoolScratchBuf,
  void* workspaceBuf,
  size_t workspaceBytes
) const {
  aclDataType dtype = useFP16 ? ACL_FLOAT16 : ACL_FLOAT;

  // Match CUDA trunk flow:
  // 1. initialConv -> scratchBuf (not trunkOutputBuf!)
  // 2. initialMatMul -> trunkOutputBuf, broadcast-add into scratchBuf
  // 3. Residual blocks ping-pong between trunkOutputBuf and scratchBuf
  // 4. trunkTipBN: scratchBuf -> trunkOutputBuf

  // Step 1: Initial conv -> scratchBuf
  initialConv->apply(handle, stream, batchSize, nnXLen, nnYLen, false, inputBuf, scratchBuf, workspaceBuf, workspaceBytes);

  // Step 2: Initial matmul -> trunkOutputBuf
  initialMatMul->apply(handle, stream, batchSize, inputGlobalBuf, trunkOutputBuf, workspaceBuf, workspaceBytes);

  // Broadcast-add: scratchBuf += trunkOutputBuf reshaped as (N, C, 1, 1)
  {
    vector<int64_t> scratchShape = {batchSize, trunkNumChannels, nnYLen, nnXLen};
    vector<int64_t> biasShape = {batchSize, trunkNumChannels, 1, 1};

    aclTensor* scratchTensor = handle->tensorCache.get(scratchBuf, scratchShape, dtype, ACL_FORMAT_NCHW);
    aclTensor* biasTensor = handle->tensorCache.get(trunkOutputBuf, biasShape, dtype, ACL_FORMAT_NCHW);
    aclTensor* resultTensor = handle->tensorCache.get(scratchBuf, scratchShape, dtype, ACL_FORMAT_NCHW);

    uint64_t addWsSize = 0;
    aclOpExecutor* addExecutor = nullptr;
    aclnnStatus status = aclnnAddGetWorkspaceSize(scratchTensor, biasTensor, handle->alphaOneScalar, resultTensor, &addWsSize, &addExecutor);
    if(status == ACLNN_SUCCESS) {
      status = aclnnAdd(workspaceBuf, addWsSize, addExecutor, stream);
    }
    if(status != ACLNN_SUCCESS) {
      throw StringError("aclnnAdd failed for initial global features: " + to_string(status));
    }
  }

  // TODO: Handle metadata features if present
  (void)inputMetaBuf;
  (void)hasMetadataEncoder;
  (void)metaMul1;
  (void)metaBias1;
  (void)metaMul2;
  (void)metaBias2;
  (void)metaMul3;

  // Step 3: Trunk blocks (Residual + GlobalPooling) - iterated in model order
  // Match CUDA's BlockStack: iterate blocks in their original order,
  // using per-type indices since blocks are stored in separate vectors.
  {
    int regularIdx = 0;
    int gpoolIdx = 0;
    int nestedIdx = 0;
    for(int i = 0; i < (int)trunkBlockKinds.size(); i++) {
      switch(trunkBlockKinds[i]) {
      case ORDINARY_BLOCK_KIND:
        residualBlocks[regularIdx]->apply(
          handle, stream, batchSize, nnXLen, nnYLen, maskBuf,
          scratchBuf, trunkOutputBuf, residualMidBuf,
          workspaceBuf, workspaceBytes);
        regularIdx++;
        break;
      case GLOBAL_POOLING_BLOCK_KIND:
        gpoolBlocks[gpoolIdx]->apply(
          handle, stream, batchSize, maskBuf, maskSumBuf,
          scratchBuf, trunkOutputBuf, residualMidBuf,
          gpoolConvOutBuf, gpoolScratchBuf,
          workspaceBuf, workspaceBytes);
        gpoolIdx++;
        break;
      case NESTED_BOTTLENECK_BLOCK_KIND:
        nestedBlocks[nestedIdx]->apply(
          handle, stream, batchSize, maskBuf, maskSumBuf,
          scratchBuf, trunkOutputBuf, residualMidBuf,
          gpoolConvOutBuf, gpoolScratchBuf,
          workspaceBuf, workspaceBytes);
        nestedIdx++;
        break;
      default:
        throw StringError("Unknown trunk block kind: " + to_string(trunkBlockKinds[i]));
      }
    }
  }

  // Step 4: Trunk tip BN: scratchBuf -> trunkOutputBuf
  trunkTipBN->apply(handle, stream, batchSize, scratchBuf, maskBuf, trunkOutputBuf, workspaceBuf, workspaceBytes);
}

void Model::applyPolicyHead(
  ComputeHandle* handle,
  aclrtStream stream,
  int batchSize,
  const void* trunkOutputBuf,
  const void* maskBuf,
  const void* maskFloatBuf,
  float* maskSumBuf,
  float* policyPassBuf,
  float* policyBuf,
  void* scratchBuf,
  void* workspaceBuf,
  size_t workspaceBytes,
  Buffers* buffers
) const {
  // Match CUDA policy head exactly:
  // 1. p1Conv: trunk -> p1Out (FP16 or FP32)
  // 2. g1Conv: trunk -> g1Out (FP16 or FP32)
  // 3. g1BN: g1Out -> g1Out2 (FP16 or FP32)
  // 4. Global pooling: g1Out2 -> g1Concat (ALWAYS FP32)
  // 5. gpoolToBiasMul: g1Concat -> g1Bias (ALWAYS FP32)
  // 6. Convert p1Out to FP32 if needed, add g1Bias (ALWAYS FP32)
  // 7. p1BN: FP32 -> scratch (ALWAYS FP32)
  // 8. p2Conv: scratch -> policyBuf (ALWAYS FP32)
  // 9. gpoolToPassMul/PassBias/PassMul2 -> policyPassBuf (ALWAYS FP32)

  aclDataType dtype = useFP16 ? ACL_FLOAT16 : ACL_FLOAT;
  int p1Channels = p1Conv->outChannels;
  int g1Channels = g1Conv->outChannels;

  void* p1OutBuf = buffers->p1OutBuf;          // ALWAYS float-sized
  void* g1OutBuf = buffers->g1OutBuf;
  void* g1Out2Buf = buffers->g1Out2Buf;
  void* g1ConcatBuf = buffers->g1ConcatBuf;    // ALWAYS float
  void* g1BiasBuf = buffers->g1BiasBuf;        // ALWAYS float
  void* p1PassBuf = buffers->p1PassBuf;        // ALWAYS float

  (void)maskSumBuf;  // TODO: Use for mask-aware pooling

  // Step 1: Apply p1Conv: trunk -> p1Out
  if(useFP16) {
    // p1Conv outputs FP16, p1OutBuf is float-sized, so we need FP16->FP32 conversion
    // Use host-side conversion (aclnnCast is broken on some CANN/hardware combos)
    p1Conv->apply(handle, stream, batchSize, nnXLen, nnYLen, false, const_cast<void*>(trunkOutputBuf), scratchBuf, workspaceBuf, workspaceBytes);
    size_t p1Elts = (size_t)batchSize * p1Channels * nnYLen * nnXLen;
    castDeviceFP16ToFP32(stream, p1OutBuf, scratchBuf, p1Elts);
  } else {
    p1Conv->apply(handle, stream, batchSize, nnXLen, nnYLen, false, const_cast<void*>(trunkOutputBuf), p1OutBuf, workspaceBuf, workspaceBytes);
  }
  // Now p1OutBuf always contains FP32 data

  // Step 2: Apply g1Conv: trunk -> g1Out
  g1Conv->apply(handle, stream, batchSize, nnXLen, nnYLen, false, const_cast<void*>(trunkOutputBuf), g1OutBuf, workspaceBuf, workspaceBytes);

  // Step 3: Apply g1BN: g1Out -> g1Out2
  g1BN->apply(handle, stream, batchSize, g1OutBuf, maskBuf, g1Out2Buf, workspaceBuf, workspaceBytes);

  // Step 4: Global pooling on g1Out2 -> g1Concat (ALWAYS FP32)
  // Computes: [mean, scaledMean, max] -> [batch, g1Channels * 3]
  {
    size_t poolElts = (size_t)batchSize * g1Channels;
    size_t poolBytesFP32 = poolElts * sizeof(float);
    size_t poolBytesDtype = useFP16 ? (poolElts * sizeof(aclFloat16)) : poolBytesFP32;

    // Use scratchBuf for intermediate pooling results
    void* meanPoolBuf = scratchBuf;
    void* meanFP32Buf = (char*)scratchBuf + poolBytesDtype;
    void* maxPoolBuf = (char*)meanFP32Buf + poolBytesFP32;
    void* maxFP32Buf = (char*)maxPoolBuf + poolBytesDtype;
    void* scaledMeanBuf = (char*)maxFP32Buf + poolBytesFP32;

    // 4a: Mean pooling (FP16 input -> FP16/FP32 output)
    aclTensor* g1Out2TensorND = handle->tensorCache.get(g1Out2Buf, {batchSize, g1Channels, nnYLen, nnXLen}, dtype, ACL_FORMAT_ND);
    aclTensor* meanPoolTensorND = handle->tensorCache.get(meanPoolBuf, {batchSize, g1Channels, 1, 1}, dtype, ACL_FORMAT_ND);

    aclIntArray* meanReduceDims = createAclIntArray({2, 3});
    uint64_t meanWsSize = 0;
    aclOpExecutor* meanExecutor = nullptr;
    aclnnStatus status = aclnnMeanGetWorkspaceSize(g1Out2TensorND, meanReduceDims, true, dtype, meanPoolTensorND, &meanWsSize, &meanExecutor);
    if(status == ACLNN_SUCCESS && meanWsSize <= workspaceBytes) {
      status = aclnnMean(workspaceBuf, meanWsSize, meanExecutor, stream);
    }
    aclDestroyIntArray(meanReduceDims);
    if(status != ACLNN_SUCCESS) {
      throw StringError("aclnnMean failed for policy head mean pooling: " + to_string(status));
    }

    // Cast mean to FP32 if needed
    aclTensor* meanFP32Tensor;
    if(useFP16) {
      aclTensor* srcTensor = handle->tensorCache.get(meanPoolBuf, {batchSize, g1Channels, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
      meanFP32Tensor = handle->tensorCache.get(meanFP32Buf, {batchSize, g1Channels, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
      uint64_t castWsSize = 0;
      aclOpExecutor* castExecutor = nullptr;
      status = aclnnCastGetWorkspaceSize(srcTensor, ACL_FLOAT, meanFP32Tensor, &castWsSize, &castExecutor);
      if(status == ACLNN_SUCCESS && castWsSize <= workspaceBytes) {
        status = aclnnCast(workspaceBuf, castWsSize, castExecutor, stream);
      }
      if(status != ACLNN_SUCCESS) {
        throw StringError("aclnnCast failed for policy mean FP16->FP32: " + to_string(status));
      }
    } else {
      meanFP32Tensor = handle->tensorCache.get(meanPoolBuf, {batchSize, g1Channels, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
      meanFP32Buf = meanPoolBuf;
    }

    // 4b: Max pooling
    aclTensor* maxPoolTensor = handle->tensorCache.get(maxPoolBuf, {batchSize, g1Channels, 1, 1}, dtype, ACL_FORMAT_ND);
    aclIntArray* reduceDims = createAclIntArray({2, 3});
    uint64_t maxWsSize = 0;
    aclOpExecutor* maxExecutor = nullptr;
    status = aclnnAmaxGetWorkspaceSize(g1Out2TensorND, reduceDims, true, maxPoolTensor, &maxWsSize, &maxExecutor);
    if(status == ACLNN_SUCCESS && maxWsSize <= workspaceBytes) {
      status = aclnnAmax(workspaceBuf, maxWsSize, maxExecutor, stream);
    }
    aclDestroyIntArray(reduceDims);
    if(status != ACLNN_SUCCESS) {
      throw StringError("aclnnAmax failed for policy head max pooling: " + to_string(status));
    }

    // Cast max to FP32 if needed
    aclTensor* maxFP32Tensor;
    if(useFP16) {
      aclTensor* srcTensor = handle->tensorCache.get(maxPoolBuf, {batchSize, g1Channels, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
      maxFP32Tensor = handle->tensorCache.get(maxFP32Buf, {batchSize, g1Channels, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
      uint64_t castWsSize = 0;
      aclOpExecutor* castExecutor = nullptr;
      status = aclnnCastGetWorkspaceSize(srcTensor, ACL_FLOAT, maxFP32Tensor, &castWsSize, &castExecutor);
      if(status == ACLNN_SUCCESS && castWsSize <= workspaceBytes) {
        status = aclnnCast(workspaceBuf, castWsSize, castExecutor, stream);
      }
      if(status != ACLNN_SUCCESS) {
        throw StringError("aclnnCast failed for policy max FP16->FP32: " + to_string(status));
      }
    } else {
      maxFP32Tensor = handle->tensorCache.get(maxPoolBuf, {batchSize, g1Channels, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
      maxFP32Buf = maxPoolBuf;
    }

    // 4c: scaledMean = mean * (sqrt(area) - 14) * 0.1
    float sqrtArea = sqrtf((float)(nnXLen * nnYLen));
    float scale = (sqrtArea - 14.0f) * 0.1f;
    aclTensor* scaledMeanTensor = handle->tensorCache.get(scaledMeanBuf, {batchSize, g1Channels, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    {
      aclScalar* scaleScalar = aclCreateScalar(&scale, ACL_FLOAT);
      uint64_t mulsWsSize = 0;
      aclOpExecutor* mulsExecutor = nullptr;
      status = aclnnMulsGetWorkspaceSize(meanFP32Tensor, scaleScalar, scaledMeanTensor, &mulsWsSize, &mulsExecutor);
      if(status == ACLNN_SUCCESS && mulsWsSize <= workspaceBytes) {
        status = aclnnMuls(workspaceBuf, mulsWsSize, mulsExecutor, stream);
      }
      aclDestroyScalar(scaleScalar);
      if(status != ACLNN_SUCCESS) {
        throw StringError("aclnnMuls failed for policy scaledMean: " + to_string(status));
      }
    }

    // 4d: Concatenate [mean, scaledMean, max] -> g1Concat (FP32) using D2D copies
    {
      size_t segmentBytes = poolElts * sizeof(float);
      aclError copyErr = aclrtMemcpyAsync(g1ConcatBuf, segmentBytes, meanFP32Buf, segmentBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
      if(copyErr != ACL_SUCCESS)
        throw StringError("aclrtMemcpyAsync failed for policy concat segment 0: " + to_string(copyErr));
      copyErr = aclrtMemcpyAsync((char*)g1ConcatBuf + segmentBytes, segmentBytes, scaledMeanBuf, segmentBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
      if(copyErr != ACL_SUCCESS)
        throw StringError("aclrtMemcpyAsync failed for policy concat segment 1: " + to_string(copyErr));
      copyErr = aclrtMemcpyAsync((char*)g1ConcatBuf + 2 * segmentBytes, segmentBytes, maxFP32Buf, segmentBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
      if(copyErr != ACL_SUCCESS)
        throw StringError("aclrtMemcpyAsync failed for policy concat segment 2: " + to_string(copyErr));
    }
  }

  // Step 5: gpoolToBiasMul: g1Concat (FP32) -> g1Bias (FP32)
  // gpoolToBiasMul is constructed with useFP16=false, so it does FP32 matmul
  gpoolToBiasMul->apply(handle, stream, batchSize, g1ConcatBuf, g1BiasBuf, workspaceBuf, workspaceBytes);

  // Step 6: Add g1Bias (FP32) to p1Out (FP32) - broadcast across spatial dims
  {
    vector<int64_t> p1OutShape = {batchSize, p1Channels, nnYLen, nnXLen};
    vector<int64_t> biasShape = {batchSize, p1Channels, 1, 1};

    aclTensor* p1OutTensor = handle->tensorCache.get(p1OutBuf, p1OutShape, ACL_FLOAT, ACL_FORMAT_NCHW);
    aclTensor* biasTensor = handle->tensorCache.get(g1BiasBuf, biasShape, ACL_FLOAT, ACL_FORMAT_NCHW);
    aclTensor* resultTensor = handle->tensorCache.get(p1OutBuf, p1OutShape, ACL_FLOAT, ACL_FORMAT_NCHW);

    uint64_t addWsSize = 0;
    aclOpExecutor* addExecutor = nullptr;
    aclnnStatus status = aclnnAddGetWorkspaceSize(p1OutTensor, biasTensor, handle->alphaOneScalar, resultTensor, &addWsSize, &addExecutor);
    if(status == ACLNN_SUCCESS && addWsSize <= workspaceBytes) {
      status = aclnnAdd(workspaceBuf, addWsSize, addExecutor, stream);
    }
    if(status != ACLNN_SUCCESS) {
      throw StringError("aclnnAdd failed for policy head bias: " + to_string(status));
    }
  }

  // Step 7: p1BN: p1Out (FP32) -> scratchBuf (FP32)
  // p1BN is constructed with useFP16=false, so it does FP32 BN
  // CRITICAL: p1BN uses FP32 dtype internally, so it needs FP32 mask (maskFloatBuf),
  // not the model-dtype maskBuf (which is FP16 in FP16 mode)
  p1BN->apply(handle, stream, batchSize, p1OutBuf, maskFloatBuf, scratchBuf, workspaceBuf, workspaceBytes);

  // Step 8: p2Conv: scratchBuf (FP32) -> policyBuf (FP32)
  // p2Conv is constructed with useFP16=false
  p2Conv->apply(handle, stream, batchSize, nnXLen, nnYLen, false, scratchBuf, policyBuf, workspaceBuf, workspaceBytes);

  // Step 9: Policy pass logit (ALWAYS FP32)
  // gpoolToPassMul, gpoolToPassBias, gpoolToPassMul2 all useFP16=false
  if(modelVersion >= 15) {
    gpoolToPassMul->apply(handle, stream, batchSize, g1ConcatBuf, p1PassBuf, workspaceBuf, workspaceBytes);
    gpoolToPassBias->apply(handle, stream, batchSize, p1PassBuf, p1PassBuf, workspaceBuf, workspaceBytes);
    gpoolToPassMul2->apply(handle, stream, batchSize, p1PassBuf, policyPassBuf, workspaceBuf, workspaceBytes);
  } else {
    gpoolToPassMul->apply(handle, stream, batchSize, g1ConcatBuf, policyPassBuf, workspaceBuf, workspaceBytes);
  }

  (void)dtype;
}

void Model::applyValueHead(
  ComputeHandle* handle,
  aclrtStream stream,
  int batchSize,
  const void* trunkOutputBuf,
  const void* maskBuf,
  float* maskSumBuf,
  float* valueBuf,
  float* scoreValueBuf,
  void* ownershipBuf,
  void* scratchBuf,
  void* workspaceBuf,
  size_t workspaceBytes,
  Buffers* buffers
) const {
  // Match CUDA value head exactly:
  // 1. v1Conv: trunk -> v1Out (FP16 or FP32)
  // 2. v1BN: v1Out -> v1Out2 (FP16 or FP32)
  // 3. Convert v1Out2 to FP32 if FP16
  // 4. Global pooling: -> v1Mean (ALWAYS FP32) [mean, scaledMean1, scaledMean2]
  // 5. v2Mul: v1Mean -> v2Out (ALWAYS FP32)
  // 6. v2Bias: v2Out -> v2Out (ALWAYS FP32)
  // 7. v3Mul+v3Bias: v2Out -> valueBuf (ALWAYS FP32)
  // 8. sv3Mul+sv3Bias: v2Out -> scoreValueBuf (ALWAYS FP32)
  // 9. vOwnershipConv: v1Out2 -> ownershipBuf (FP16 conv -> FP32 output)

  aclDataType dtype = useFP16 ? ACL_FLOAT16 : ACL_FLOAT;
  int v1Channels = v1Conv->outChannels;
  int ownershipChannels = vOwnershipConv->outChannels;

  void* v1OutBuf = buffers->v1OutBuf;
  void* v1Out2Buf = buffers->v1Out2Buf;
  void* v1MeanBuf = buffers->v1MeanBuf;      // ALWAYS float
  void* v2OutBuf = buffers->v2OutBuf;        // ALWAYS float
  void* ownershipScratchBuf = buffers->ownershipScratchBuf;

  (void)maskSumBuf;  // TODO: Use for mask-aware pooling

  // Step 1: Apply v1Conv: trunk -> v1Out
  v1Conv->apply(handle, stream, batchSize, nnXLen, nnYLen, false, const_cast<void*>(trunkOutputBuf), v1OutBuf, workspaceBuf, workspaceBytes);

  // Step 2: Apply v1BN: v1Out -> v1Out2
  v1BN->apply(handle, stream, batchSize, v1OutBuf, maskBuf, v1Out2Buf, workspaceBuf, workspaceBytes);

  // Step 3: Value head global pooling on v1Out2 -> v1Mean (ALWAYS FP32)
  // Value head: [mean, scaledMean1, scaledMean2] (NO max)
  {
    size_t poolElts = (size_t)batchSize * v1Channels;
    size_t poolBytesFP32 = poolElts * sizeof(float);
    size_t poolBytesDtype = useFP16 ? (poolElts * sizeof(aclFloat16)) : poolBytesFP32;

    void* meanPoolBuf = scratchBuf;
    void* meanFP32Buf = (char*)scratchBuf + poolBytesDtype;

    // 3a: Mean pooling
    aclTensor* v1Out2TensorND = handle->tensorCache.get(v1Out2Buf, {batchSize, v1Channels, nnYLen, nnXLen}, dtype, ACL_FORMAT_ND);
    aclTensor* meanPoolTensorND = handle->tensorCache.get(meanPoolBuf, {batchSize, v1Channels, 1, 1}, dtype, ACL_FORMAT_ND);

    aclIntArray* meanReduceDims = createAclIntArray({2, 3});
    uint64_t meanWsSize = 0;
    aclOpExecutor* meanExecutor = nullptr;
    aclnnStatus status = aclnnMeanGetWorkspaceSize(v1Out2TensorND, meanReduceDims, true, dtype, meanPoolTensorND, &meanWsSize, &meanExecutor);
    if(status == ACLNN_SUCCESS && meanWsSize <= workspaceBytes) {
      status = aclnnMean(workspaceBuf, meanWsSize, meanExecutor, stream);
    }
    aclDestroyIntArray(meanReduceDims);
    if(status != ACLNN_SUCCESS) {
      throw StringError("aclnnMean failed for value head mean pooling: " + to_string(status));
    }

    // 3b: Cast mean to FP32 if needed
    aclTensor* meanFP32Tensor;
    if(useFP16) {
      aclTensor* srcTensor = handle->tensorCache.get(meanPoolBuf, {batchSize, v1Channels, 1, 1}, ACL_FLOAT16, ACL_FORMAT_ND);
      meanFP32Tensor = handle->tensorCache.get(meanFP32Buf, {batchSize, v1Channels, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
      uint64_t castWsSize = 0;
      aclOpExecutor* castExecutor = nullptr;
      status = aclnnCastGetWorkspaceSize(srcTensor, ACL_FLOAT, meanFP32Tensor, &castWsSize, &castExecutor);
      if(status == ACLNN_SUCCESS && castWsSize <= workspaceBytes) {
        status = aclnnCast(workspaceBuf, castWsSize, castExecutor, stream);
      }
      if(status != ACLNN_SUCCESS) {
        throw StringError("aclnnCast failed for value mean FP16->FP32: " + to_string(status));
      }
    } else {
      meanFP32Tensor = handle->tensorCache.get(meanPoolBuf, {batchSize, v1Channels, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
      meanFP32Buf = meanPoolBuf;
    }

    // 3c: scaledMean1 = mean * (sqrt(area) - 14) * 0.1
    float sqrtArea = sqrtf((float)(nnXLen * nnYLen));
    float scale1 = (sqrtArea - 14.0f) * 0.1f;

    void* scaledMean1Buf = (char*)meanFP32Buf + poolBytesFP32;
    aclTensor* scaledMean1Tensor = handle->tensorCache.get(scaledMean1Buf, {batchSize, v1Channels, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    {
      aclScalar* scaleScalar = aclCreateScalar(&scale1, ACL_FLOAT);
      uint64_t mulsWsSize = 0;
      aclOpExecutor* mulsExecutor = nullptr;
      status = aclnnMulsGetWorkspaceSize(meanFP32Tensor, scaleScalar, scaledMean1Tensor, &mulsWsSize, &mulsExecutor);
      if(status == ACLNN_SUCCESS && mulsWsSize <= workspaceBytes) {
        status = aclnnMuls(workspaceBuf, mulsWsSize, mulsExecutor, stream);
      }
      aclDestroyScalar(scaleScalar);
      if(status != ACLNN_SUCCESS) {
        throw StringError("aclnnMuls failed for value scaledMean1: " + to_string(status));
      }
    }

    // 3d: scaledMean2 = mean * ((sqrt(area) - 14)^2 * 0.01 - 0.1)
    float scale2 = (sqrtArea - 14.0f) * (sqrtArea - 14.0f) * 0.01f - 0.1f;

    void* scaledMean2Buf = (char*)scaledMean1Buf + poolBytesFP32;
    aclTensor* scaledMean2Tensor = handle->tensorCache.get(scaledMean2Buf, {batchSize, v1Channels, 1, 1}, ACL_FLOAT, ACL_FORMAT_ND);
    {
      aclScalar* scaleScalar = aclCreateScalar(&scale2, ACL_FLOAT);
      uint64_t mulsWsSize = 0;
      aclOpExecutor* mulsExecutor = nullptr;
      status = aclnnMulsGetWorkspaceSize(meanFP32Tensor, scaleScalar, scaledMean2Tensor, &mulsWsSize, &mulsExecutor);
      if(status == ACLNN_SUCCESS && mulsWsSize <= workspaceBytes) {
        status = aclnnMuls(workspaceBuf, mulsWsSize, mulsExecutor, stream);
      }
      aclDestroyScalar(scaleScalar);
      if(status != ACLNN_SUCCESS) {
        throw StringError("aclnnMuls failed for value scaledMean2: " + to_string(status));
      }
    }

    // 3e: Concatenate [mean, scaledMean1, scaledMean2] -> v1Mean (FP32) using D2D copies
    {
      size_t segmentBytes = poolElts * sizeof(float);
      aclError copyErr = aclrtMemcpyAsync(v1MeanBuf, segmentBytes, meanFP32Buf, segmentBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
      if(copyErr != ACL_SUCCESS)
        throw StringError("aclrtMemcpyAsync failed for value concat segment 0: " + to_string(copyErr));
      copyErr = aclrtMemcpyAsync((char*)v1MeanBuf + segmentBytes, segmentBytes, scaledMean1Buf, segmentBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
      if(copyErr != ACL_SUCCESS)
        throw StringError("aclrtMemcpyAsync failed for value concat segment 1: " + to_string(copyErr));
      copyErr = aclrtMemcpyAsync((char*)v1MeanBuf + 2 * segmentBytes, segmentBytes, scaledMean2Buf, segmentBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, stream);
      if(copyErr != ACL_SUCCESS)
        throw StringError("aclrtMemcpyAsync failed for value concat segment 2: " + to_string(copyErr));
    }
  }

  // Step 4: v2Mul: v1Mean (FP32) -> v2Out (FP32)
  // v2Mul is constructed with useFP16=false
  v2Mul->apply(handle, stream, batchSize, v1MeanBuf, v2OutBuf, workspaceBuf, workspaceBytes);

  // Step 5: v2Bias: v2Out -> v2Out (in-place, ALWAYS FP32)
  v2Bias->apply(handle, stream, batchSize, v2OutBuf, v2OutBuf, workspaceBuf, workspaceBytes);

  // Step 6-7: v3Mul + v3Bias: v2Out -> valueBuf (ALWAYS FP32)
  v3Mul->apply(handle, stream, batchSize, v2OutBuf, valueBuf, workspaceBuf, workspaceBytes);
  v3Bias->apply(handle, stream, batchSize, valueBuf, valueBuf, workspaceBuf, workspaceBytes);

  // Step 8-9: sv3Mul + sv3Bias: v2Out -> scoreValueBuf (ALWAYS FP32)
  sv3Mul->apply(handle, stream, batchSize, v2OutBuf, scoreValueBuf, workspaceBuf, workspaceBytes);
  sv3Bias->apply(handle, stream, batchSize, scoreValueBuf, scoreValueBuf, workspaceBuf, workspaceBytes);

  // Step 10: vOwnershipConv: v1Out2 -> ownershipBuf
  // vOwnershipConv uses FP16, ownershipBuf is FP32
  if(useFP16) {
    vOwnershipConv->apply(handle, stream, batchSize, nnXLen, nnYLen, false, v1Out2Buf, ownershipScratchBuf, workspaceBuf, workspaceBytes);
    aclTensor* srcTensor = handle->tensorCache.get(ownershipScratchBuf, {batchSize, ownershipChannels, nnYLen, nnXLen}, ACL_FLOAT16, ACL_FORMAT_NCHW);
    aclTensor* dstTensor = handle->tensorCache.get(ownershipBuf, {batchSize, ownershipChannels, nnYLen, nnXLen}, ACL_FLOAT, ACL_FORMAT_NCHW);
    uint64_t castWsSize = 0;
    aclOpExecutor* castExecutor = nullptr;
    aclnnStatus status = aclnnCastGetWorkspaceSize(srcTensor, ACL_FLOAT, dstTensor, &castWsSize, &castExecutor);
    if(status == ACLNN_SUCCESS && castWsSize <= workspaceBytes) {
      status = aclnnCast(workspaceBuf, castWsSize, castExecutor, stream);
    }
    if(status != ACLNN_SUCCESS) {
      throw StringError("aclnnCast failed for ownership FP16->FP32: " + to_string(status));
    }
  } else {
    vOwnershipConv->apply(handle, stream, batchSize, nnXLen, nnYLen, false, v1Out2Buf, ownershipBuf, workspaceBuf, workspaceBytes);
  }

  (void)dtype;
}

//---------------------------------------------------------------------------------
// ComputeContext / ComputeHandle creation
//---------------------------------------------------------------------------------

ComputeContext* NeuralNet::createComputeContext(
  const vector<int>& gpuIdxs,
  Logger* logger,
  int nnXLen,
  int nnYLen,
  const string& openCLTunerFile,
  const string& homeDataDirOverride,
  bool openCLReTunePerBoardSize,
  enabled_t useFP16Mode,
  enabled_t useNHWCMode,
  const LoadedModel* loadedModel
) {
  (void)logger;
  (void)openCLTunerFile;
  (void)homeDataDirOverride;
  (void)openCLReTunePerBoardSize;
  (void)loadedModel;

  vector<int> actualGpuIdxs = gpuIdxs;
  if(actualGpuIdxs.size() <= 0 || (actualGpuIdxs.size() == 1 && actualGpuIdxs[0] == -1)) {
    actualGpuIdxs = {0};
  }

  // Set default device before any allocations (model weights, buffers, etc.)
  int defaultDevice = actualGpuIdxs[0];
  aclError ret = aclrtSetDevice(defaultDevice);
  if(ret != ACL_SUCCESS) {
    throw StringError("aclrtSetDevice failed in createComputeContext for device " + to_string(defaultDevice) + " with error: " + to_string(ret));
  }

  return new ComputeContext(nnXLen, nnYLen, useFP16Mode, useNHWCMode, actualGpuIdxs);
}

void NeuralNet::freeComputeContext(ComputeContext* computeContext) {
  delete computeContext;
}

ComputeHandle* NeuralNet::createComputeHandle(
  ComputeContext* context,
  const LoadedModel* loadedModel,
  Logger* logger,
  int maxBatchSize,
  bool requireExactNNLen,
  bool inputsUseNHWC,
  int gpuIdxForThisThread,
  int serverThreadIdx
) {
  (void)serverThreadIdx;

  int deviceIdx = (gpuIdxForThisThread == -1) ? 0 : gpuIdxForThisThread;

  // Determine FP16 mode - Ascend 910ProA is optimized for FP16
  // Enable by default for maximum performance
  bool useFP16 = true;
  if(context->useFP16Mode == enabled_t::True)
    useFP16 = true;
  else if(context->useFP16Mode == enabled_t::False)
    useFP16 = false;
  // else auto -> use FP16 (already set to true above)

  ComputeHandle* handle = new ComputeHandle(
    deviceIdx, useFP16, context->nnXLen, context->nnYLen, requireExactNNLen, inputsUseNHWC
  );

  // Set device and create stream
  aclError ret = aclrtSetDevice(deviceIdx);
  if(ret != ACL_SUCCESS) {
    delete handle;
    throw StringError("aclrtSetDevice failed for device " + to_string(deviceIdx) + " with error: " + to_string(ret));
  }

  ret = aclrtCreateStream(&handle->stream);
  if(ret != ACL_SUCCESS) {
    delete handle;
    throw StringError("aclrtCreateStream failed with error: " + to_string(ret));
  }

  // Initialize cached scalars (alpha=1.0 for Add operations)
  handle->initScalars();

  // Log device assignment for multi-NPU debugging
  if(logger != nullptr) {
    logger->write(
      "Ascend NPU backend thread " + Global::intToString(serverThreadIdx)
      + ": using device " + Global::intToString(deviceIdx)
      + ", FP16 " + string(useFP16 ? "enabled" : "disabled")
    );
  }

  // Create model
  const ModelDesc& modelDesc = loadedModel->modelDesc;
  Model* model = new Model(modelDesc, context->nnXLen, context->nnYLen, useFP16);
  handle->model = model;

  // Create scratch buffers
  size_t workspaceNeeded = model->requiredWorkspaceBytes(maxBatchSize);
  // Ensure minimum workspace for multi-NPU stability - 256MB minimum
  // This prevents resource exhaustion when many ops are queued simultaneously
  size_t minWorkspace = 256 * 1024 * 1024;
  workspaceNeeded = std::max(workspaceNeeded, minWorkspace);
  ScratchBuffers* scratch = new ScratchBuffers(maxBatchSize, context->nnXLen, context->nnYLen, useFP16, workspaceNeeded);
  handle->scratch = scratch;

  // Create device buffers
  Buffers* buffers = new Buffers(modelDesc, maxBatchSize, context->nnXLen, context->nnYLen, useFP16, workspaceNeeded);
  handle->buffers = buffers;

  return handle;
}

void NeuralNet::freeComputeHandle(ComputeHandle* computeHandle) {
  if(computeHandle != nullptr) {
    // Set device context before freeing resources on the correct device
    aclrtSetDevice(computeHandle->deviceIdx);
    delete computeHandle->buffers;
    delete computeHandle->scratch;
    delete computeHandle->model;
    delete computeHandle;
  }
}

bool NeuralNet::isUsingFP16(const ComputeHandle* computeHandle) {
  return computeHandle->usingFP16;
}

//---------------------------------------------------------------------------------
// getOutput
//---------------------------------------------------------------------------------

void NeuralNet::getOutput(
  ComputeHandle* gpuHandle,
  InputBuffers* inputBuffers,
  int numBatchEltsFilled,
  NNResultBuf** inputBufs,
  vector<NNOutput*>& outputs
) {
  // CRITICAL: Guard against segfault when server threads > search threads
  if(numBatchEltsFilled <= 0) {
    return;
  }
  // Set device context - critical for multi-NPU since each server thread
  // needs to be bound to its device for all ACL/ACLNN calls
  aclrtSetDevice(gpuHandle->deviceIdx);

  assert(numBatchEltsFilled <= inputBuffers->maxBatchSize);
  const int batchSize = numBatchEltsFilled;
  const int nnXLen = gpuHandle->nnXLen;
  const int nnYLen = gpuHandle->nnYLen;
  const int modelVersion = gpuHandle->model->modelVersion;

  const int numSpatialFeatures = NNModelVersion::getNumSpatialFeatures(modelVersion);
  const int numGlobalFeatures = NNModelVersion::getNumGlobalFeatures(modelVersion);
  const int numMetaFeatures = (int)inputBuffers->singleInputMetaElts;
  assert(numSpatialFeatures == gpuHandle->model->numInputChannels);
  assert(numSpatialFeatures * nnXLen * nnYLen == inputBuffers->singleInputElts);
  assert(numGlobalFeatures == inputBuffers->singleInputGlobalElts);
  const int numPolicyChannels = gpuHandle->model->numPolicyChannels;

  // Copy inputs from individual NNResultBufs to batched host buffers
  for(int nIdx = 0; nIdx < batchSize; nIdx++) {
    float* rowSpatialInput = inputBuffers->userInputBuffer + (inputBuffers->singleInputElts * nIdx);
    float* rowGlobalInput = inputBuffers->userInputGlobalBuffer + (inputBuffers->singleInputGlobalElts * nIdx);
    float* rowMetaInput = inputBuffers->userInputMetaBuffer + (inputBuffers->singleInputMetaElts * nIdx);

    const float* rowGlobal = inputBufs[nIdx]->rowGlobalBuf.data();
    const float* rowSpatial = inputBufs[nIdx]->rowSpatialBuf.data();
    const float* rowMeta = inputBufs[nIdx]->rowMetaBuf.data();
    bool hasRowMeta = inputBufs[nIdx]->hasRowMeta;

    std::copy(rowGlobal, rowGlobal + numGlobalFeatures, rowGlobalInput);

    if(numMetaFeatures > 0) {
      testAssert(rowMeta != nullptr);
      testAssert(hasRowMeta);
      std::copy(rowMeta, rowMeta + numMetaFeatures, rowMetaInput);
    } else {
      testAssert(!hasRowMeta);
    }

    SymmetryHelpers::copyInputsWithSymmetry(
      rowSpatial, rowSpatialInput, 1, nnYLen, nnXLen, numSpatialFeatures,
      gpuHandle->inputsUseNHWC, inputBufs[nIdx]->symmetry
    );
  }

  Buffers* buffers = gpuHandle->buffers;

  // ========================================================================
  // PHASE 1: Copy inputs H2D (OUTSIDE graph capture - data changes every call)
  // ========================================================================
  if(!gpuHandle->usingFP16) {
    ascendCopyH2D(buffers->inputBuf, inputBuffers->userInputBuffer, inputBuffers->singleInputBytes * batchSize);
    ascendCopyH2D(buffers->inputGlobalBuf, inputBuffers->userInputGlobalBuffer, inputBuffers->singleInputGlobalBytes * batchSize);
    if(numMetaFeatures > 0) {
      ascendCopyH2D(buffers->inputMetaBuf, inputBuffers->userInputMetaBuffer, inputBuffers->singleInputMetaBytes * batchSize);
    }
  } else {
    // For FP16 mode, convert FP32 -> FP16 on HOST, then copy to device
    // (aclnnCast is unreliable on some CANN/hardware combinations, error 561103)
    // This is fast since input sizes are small relative to model weights

    // Convert spatial input: FP32 host -> FP16 host -> device
    {
      size_t numSpatialElts = (size_t)batchSize * numSpatialFeatures * nnYLen * nnXLen;
      const float* fp32Data = (const float*)inputBuffers->userInputBuffer;
      vector<aclFloat16> fp16Data(numSpatialElts);
      for(size_t i = 0; i < numSpatialElts; i++) {
        fp16Data[i] = aclFloatToFloat16(fp32Data[i]);
      }
      ascendCopyH2D(buffers->inputBuf, fp16Data.data(), numSpatialElts * sizeof(aclFloat16));
    }

    // Convert global input
    {
      size_t numGlobalElts = (size_t)batchSize * numGlobalFeatures;
      const float* fp32Data = (const float*)inputBuffers->userInputGlobalBuffer;
      vector<aclFloat16> fp16Data(numGlobalElts);
      for(size_t i = 0; i < numGlobalElts; i++) {
        fp16Data[i] = aclFloatToFloat16(fp32Data[i]);
      }
      ascendCopyH2D(buffers->inputGlobalBuf, fp16Data.data(), numGlobalElts * sizeof(aclFloat16));
    }

    // Convert meta input if present
    if(numMetaFeatures > 0) {
      size_t numMetaElts = (size_t)batchSize * numMetaFeatures;
      const float* fp32Data = (const float*)inputBuffers->userInputMetaBuffer;
      vector<aclFloat16> fp16Data(numMetaElts);
      for(size_t i = 0; i < numMetaElts; i++) {
        fp16Data[i] = aclFloatToFloat16(fp32Data[i]);
      }
      ascendCopyH2D(buffers->inputMetaBuf, fp16Data.data(), numMetaElts * sizeof(aclFloat16));
    }
  }

  // Ensure all input copies/casts complete before inference
  aclrtSynchronizeStream(gpuHandle->stream);

  // ========================================================================
  // PHASE 2: Run model inference (eager execution)
  // ========================================================================
  gpuHandle->model->apply(
    gpuHandle, gpuHandle->stream, batchSize, gpuHandle->requireExactNNLen,
    buffers->inputBuf, buffers->inputGlobalBuf, buffers->inputMetaBuf,
    buffers->policyPassBuf, buffers->policyBuf, buffers->valueBuf,
    buffers->scoreValueBuf, buffers->ownershipBuf, buffers
  );

  // Synchronize before copying results back
  // This is critical for multi-NPU stability - ensures all ops complete before D2H copies
  aclError ret = aclrtSynchronizeStream(gpuHandle->stream);
  if(ret != ACL_SUCCESS) {
    // Error 507015 typically indicates stream timeout or resource exhaustion
    // Log detailed error for debugging multi-NPU issues
    string errMsg = "aclrtSynchronizeStream failed on device " + to_string(gpuHandle->deviceIdx)
      + " with error: " + to_string((int)ret)
      + " (batchSize=" + to_string(batchSize) + ")";
    throw StringError(errMsg);
  }

  // ========================================================================
  // PHASE 3: Copy outputs D2H (OUTSIDE graph capture - need fresh data)
  // ========================================================================
  ascendCopyD2H(inputBuffers->policyPassResults, buffers->policyPassBuf, inputBuffers->singlePolicyPassResultBytes * batchSize);
  ascendCopyD2H(inputBuffers->policyResults, buffers->policyBuf, inputBuffers->singlePolicyResultBytes * batchSize);
  ascendCopyD2H(inputBuffers->valueResults, buffers->valueBuf, inputBuffers->singleValueResultBytes * batchSize);
  ascendCopyD2H(inputBuffers->scoreValueResults, buffers->scoreValueBuf, inputBuffers->singleScoreValueResultBytes * batchSize);
  ascendCopyD2H(inputBuffers->ownershipResults, buffers->ownershipBuf, inputBuffers->singleOwnershipResultBytes * batchSize);

  // Extract results into NNOutput structs
  assert(outputs.size() == batchSize);

  float policyProbsTmp[NNPos::MAX_NN_POLICY_SIZE];

  for(int row = 0; row < batchSize; row++) {
    NNOutput* output = outputs[row];
    assert(output->nnXLen == nnXLen);
    assert(output->nnYLen == nnYLen);
    float policyOptimism = (float)inputBufs[row]->policyOptimism;

    const float* policyPassSrcBuf = inputBuffers->policyPassResults + row * numPolicyChannels;
    const float* policySrcBuf = inputBuffers->policyResults + row * numPolicyChannels * nnXLen * nnYLen;
    float* policyProbs = output->policyProbs;

    // Handle policy with optimism
    if(numPolicyChannels == 2 || (numPolicyChannels == 4 && modelVersion >= 16)) {
      if(gpuHandle->inputsUseNHWC) {
        for(int i = 0; i < nnXLen * nnYLen; i++) {
          float p = policySrcBuf[i * numPolicyChannels];
          float pOpt = policySrcBuf[i * numPolicyChannels + 1];
          policyProbsTmp[i] = p + (pOpt - p) * policyOptimism;
        }
        SymmetryHelpers::copyOutputsWithSymmetry(policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyProbs[nnXLen * nnYLen] = policyPassSrcBuf[0] + (policyPassSrcBuf[1] - policyPassSrcBuf[0]) * policyOptimism;
      } else {
        for(int i = 0; i < nnXLen * nnYLen; i++) {
          float p = policySrcBuf[i];
          float pOpt = policySrcBuf[i + nnXLen * nnYLen];
          policyProbsTmp[i] = p + (pOpt - p) * policyOptimism;
        }
        SymmetryHelpers::copyOutputsWithSymmetry(policyProbsTmp, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
        policyProbs[nnXLen * nnYLen] = policyPassSrcBuf[0] + (policyPassSrcBuf[1] - policyPassSrcBuf[0]) * policyOptimism;
      }
    } else {
      assert(numPolicyChannels == 1);
      SymmetryHelpers::copyOutputsWithSymmetry(policySrcBuf, policyProbs, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
      policyProbs[nnXLen * nnYLen] = policyPassSrcBuf[0];
    }

    // Value outputs
    int numValueChannels = gpuHandle->model->numValueChannels;
    assert(numValueChannels == 3);
    output->whiteWinProb = inputBuffers->valueResults[row * numValueChannels];
    output->whiteLossProb = inputBuffers->valueResults[row * numValueChannels + 1];
    output->whiteNoResultProb = inputBuffers->valueResults[row * numValueChannels + 2];

    // Ownership
    if(output->whiteOwnerMap != nullptr) {
      const float* ownershipSrcBuf = inputBuffers->ownershipResults + row * nnXLen * nnYLen;
      assert(gpuHandle->model->numOwnershipChannels == 1);
      SymmetryHelpers::copyOutputsWithSymmetry(ownershipSrcBuf, output->whiteOwnerMap, 1, nnYLen, nnXLen, inputBufs[row]->symmetry);
    }

    // Score/value outputs based on model version
    if(modelVersion >= 9) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 6);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
      output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 4];
      output->shorttermScoreError = inputBuffers->scoreValueResults[row * numScoreValueChannels + 5];
    } else if(modelVersion >= 8) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 4);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = inputBuffers->scoreValueResults[row * numScoreValueChannels + 2];
      output->varTimeLeft = inputBuffers->scoreValueResults[row * numScoreValueChannels + 3];
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    } else if(modelVersion >= 4) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 2);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = inputBuffers->scoreValueResults[row * numScoreValueChannels + 1];
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    } else if(modelVersion >= 3) {
      int numScoreValueChannels = gpuHandle->model->numScoreValueChannels;
      assert(numScoreValueChannels == 1);
      output->whiteScoreMean = inputBuffers->scoreValueResults[row * numScoreValueChannels];
      output->whiteScoreMeanSq = output->whiteScoreMean * output->whiteScoreMean;
      output->whiteLead = output->whiteScoreMean;
      output->varTimeLeft = 0;
      output->shorttermWinlossError = 0;
      output->shorttermScoreError = 0;
    } else {
      ASSERT_UNREACHABLE;
    }
  }
}

//---------------------------------------------------------------------------------
// Test functions
//---------------------------------------------------------------------------------

bool NeuralNet::testEvaluateConv(
  const ConvLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  vector<float>& outputBuffer
) {
  (void)useNHWC; // We always use NCHW internally

  size_t numInputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->inChannels;
  size_t numOutputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->outChannels;

  if(numInputFloats != inputBuffer.size()) {
    throw StringError("testEvaluateConv: unexpected input buffer size");
  }

  // Set device
  ACL_CHECK(aclrtSetDevice(0), "aclrtSetDevice");

  // Create stream
  aclrtStream stream;
  ACL_CHECK(aclrtCreateStream(&stream), "aclrtCreateStream");

  // Create layer
  ConvLayer* convLayer = new ConvLayer(desc, useFP16);

  // Allocate device buffers
  void* deviceInput = ascendMalloc(numInputFloats * (useFP16 ? sizeof(aclFloat16) : sizeof(float)));
  void* deviceOutput = ascendMalloc(numOutputFloats * (useFP16 ? sizeof(aclFloat16) : sizeof(float)));

  // Copy input to device
  ascendCopyH2D(deviceInput, inputBuffer.data(), numInputFloats * sizeof(float));

  // Get workspace size
  size_t workspaceBytes = convLayer->requiredWorkspaceBytes(batchSize, nnXLen, nnYLen, stream);
  void* deviceWorkspace = nullptr;
  if(workspaceBytes > 0) {
    deviceWorkspace = ascendMalloc(workspaceBytes);
  }

  // Apply convolution
  convLayer->apply(nullptr, stream, batchSize, nnXLen, nnYLen, false, deviceInput, deviceOutput, deviceWorkspace, workspaceBytes);

  // Synchronize
  ACL_CHECK(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");

  // Copy output back to host
  outputBuffer.resize(numOutputFloats);
  ascendCopyD2H(outputBuffer.data(), deviceOutput, numOutputFloats * sizeof(float));

  // Cleanup
  ascendFree(deviceWorkspace);
  ascendFree(deviceInput);
  ascendFree(deviceOutput);
  delete convLayer;
  ACL_CHECK(aclrtDestroyStream(stream), "aclrtDestroyStream");

  return true;
}

bool NeuralNet::testEvaluateBatchNorm(
  const BatchNormLayerDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  (void)useNHWC; // We always use NCHW internally

  size_t numInputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->numChannels;
  size_t numMaskFloats = (size_t)batchSize * nnXLen * nnYLen;
  size_t numOutputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->numChannels;

  if(numInputFloats != inputBuffer.size()) {
    throw StringError("testEvaluateBatchNorm: unexpected input buffer size");
  }
  if(numMaskFloats != maskBuffer.size()) {
    throw StringError("testEvaluateBatchNorm: unexpected mask buffer size");
  }

  // Set device
  ACL_CHECK(aclrtSetDevice(0), "aclrtSetDevice");

  // Create stream
  aclrtStream stream;
  ACL_CHECK(aclrtCreateStream(&stream), "aclrtCreateStream");

  // Create activation descriptor (identity for testing)
  ActivationLayerDesc actDesc;
  actDesc.activation = ACTIVATION_IDENTITY;

  // Allocate device buffers
  aclDataType dtype = useFP16 ? ACL_FLOAT16 : ACL_FLOAT;
  size_t inputBytes = numInputFloats * (useFP16 ? sizeof(aclFloat16) : sizeof(float));
  size_t maskBytes = numMaskFloats * (useFP16 ? sizeof(aclFloat16) : sizeof(float));
  size_t outputBytes = numOutputFloats * (useFP16 ? sizeof(aclFloat16) : sizeof(float));

  void* deviceInput = ascendMalloc(inputBytes);
  void* deviceMask = ascendMalloc(maskBytes);
  void* deviceOutput = ascendMalloc(outputBytes);

  // Copy inputs to device (with FP16 conversion if needed)
  if(useFP16) {
    ascendCopyH2D(deviceInput, inputBuffer.data(), numInputFloats * sizeof(float));
    ascendCopyH2D(deviceMask, maskBuffer.data(), numMaskFloats * sizeof(float));
  } else {
    ascendCopyH2D(deviceInput, inputBuffer.data(), inputBytes);
    ascendCopyH2D(deviceMask, maskBuffer.data(), maskBytes);
  }

  // Create BatchNormLayer
  BatchNormLayer* batchNormLayer = new BatchNormLayer(desc, &actDesc, nnXLen, nnYLen, useFP16);

  // BatchNormLayer doesn't need workspace in our implementation
  void* deviceWorkspace = nullptr;
  size_t workspaceBytes = 0;

  // Apply batch norm
  batchNormLayer->apply(nullptr, stream, batchSize, deviceInput, deviceMask, deviceOutput, deviceWorkspace, workspaceBytes);

  // Synchronize
  ACL_CHECK(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");

  // Copy output back to host
  outputBuffer.resize(numOutputFloats);
  ascendCopyD2H(outputBuffer.data(), deviceOutput, numOutputFloats * sizeof(float));

  // Cleanup
  ascendFree(deviceWorkspace);
  ascendFree(deviceInput);
  ascendFree(deviceMask);
  ascendFree(deviceOutput);
  delete batchNormLayer;
  ACL_CHECK(aclrtDestroyStream(stream), "aclrtDestroyStream");

  return true;
}

bool NeuralNet::testEvaluateResidualBlock(
  const ResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  (void)useNHWC; // We always use NCHW internally

  size_t numInputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->preBN.numChannels;
  size_t numMaskFloats = (size_t)batchSize * nnXLen * nnYLen;
  size_t numOutputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->finalConv.outChannels;

  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateResidualBlock: unexpected input buffer size");
  if(numMaskFloats != maskBuffer.size())
    throw StringError("testEvaluateResidualBlock: unexpected mask buffer size");

  // Set device
  ACL_CHECK(aclrtSetDevice(0), "aclrtSetDevice");

  // Create stream
  aclrtStream stream;
  ACL_CHECK(aclrtCreateStream(&stream), "aclrtCreateStream");

  aclDataType dtype = useFP16 ? ACL_FLOAT16 : ACL_FLOAT;
  size_t eltSize = useFP16 ? sizeof(aclFloat16) : sizeof(float);

  // Allocate device buffers (3 buffers for trunkBuf, trunkScratchBuf, midBuf pattern)
  void* deviceInput = ascendMalloc(numInputFloats * eltSize);
  void* deviceMask = ascendMalloc(numMaskFloats * eltSize);
  void* deviceScratch = ascendMalloc(numInputFloats * eltSize);
  void* deviceMid = ascendMalloc(numInputFloats * eltSize);

  // Copy inputs to device
  ascendCopyH2D(deviceInput, inputBuffer.data(), numInputFloats * sizeof(float));
  ascendCopyH2D(deviceMask, maskBuffer.data(), numMaskFloats * sizeof(float));

  // Create ResidualBlock
  ResidualBlock* residualBlock = new ResidualBlock(desc, nnXLen, nnYLen, useFP16);

  // Get workspace size (requires stream parameter)
  size_t workspaceBytes = residualBlock->requiredWorkspaceBytes(batchSize, nnXLen, nnYLen, stream);
  void* deviceWorkspace = nullptr;
  if(workspaceBytes > 0) {
    deviceWorkspace = ascendMalloc(workspaceBytes);
  }

  // Create scratch buffers (needs maxWorkspaceNeeded parameter)
  ScratchBuffers scratch(batchSize, nnXLen, nnYLen, useFP16, workspaceBytes);

  // Apply residual block with 3-buffer pattern (trunkBuf, trunkScratchBuf, midBuf)
  // Note: handle=nullptr means we can't use tensorCache. The apply function
  // creates tensor descriptors on the fly when handle is null (but this is not implemented
  // for the new 3-buffer pattern). For now, create a minimal handle for the test.
  // Actually, the new apply directly accesses handle->tensorCache. We need a handle.
  // Since this is a test function that creates its own stream, create a temporary handle.
  ComputeHandle* tempHandle = new ComputeHandle(0, useFP16, nnXLen, nnYLen, false, false);
  tempHandle->stream = stream;
  tempHandle->initScalars();
  // Also need model/buffers set (nullptr OK since we're not using them)
  tempHandle->model = nullptr;
  tempHandle->buffers = nullptr;

  residualBlock->apply(tempHandle, stream, batchSize, nnXLen, nnYLen, deviceMask,
                     deviceInput, deviceScratch, deviceMid, deviceWorkspace, workspaceBytes);

  // Synchronize
  ACL_CHECK(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");

  // Copy output back to host
  outputBuffer.resize(numOutputFloats);
  ascendCopyD2H(outputBuffer.data(), deviceInput, numOutputFloats * sizeof(float));

  // Cleanup
  ascendFree(deviceWorkspace);
  ascendFree(deviceInput);
  ascendFree(deviceMask);
  ascendFree(deviceScratch);
  ascendFree(deviceMid);
  delete residualBlock;
  // Clean up temp handle (stream is owned by us, don't destroy it)
  tempHandle->stream = nullptr;
  delete tempHandle;
  ACL_CHECK(aclrtDestroyStream(stream), "aclrtDestroyStream");

  return true;
}

bool NeuralNet::testEvaluateGlobalPoolingResidualBlock(
  const GlobalPoolingResidualBlockDesc* desc,
  int batchSize,
  int nnXLen,
  int nnYLen,
  bool useFP16,
  bool useNHWC,
  const vector<float>& inputBuffer,
  const vector<float>& maskBuffer,
  vector<float>& outputBuffer
) {
  (void)useNHWC; // We always use NCHW internally

  size_t numInputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->preBN.numChannels;
  size_t numMaskFloats = (size_t)batchSize * nnXLen * nnYLen;
  size_t numMaskSumFloats = (size_t)batchSize;
  size_t numOutputFloats = (size_t)batchSize * nnXLen * nnYLen * desc->finalConv.outChannels;

  if(numInputFloats != inputBuffer.size())
    throw StringError("testEvaluateGlobalPoolingResidualBlock: unexpected input buffer size");
  if(numMaskFloats != maskBuffer.size())
    throw StringError("testEvaluateGlobalPoolingResidualBlock: unexpected mask buffer size");

  // Set device
  ACL_CHECK(aclrtSetDevice(0), "aclrtSetDevice");

  // Create stream
  aclrtStream stream;
  ACL_CHECK(aclrtCreateStream(&stream), "aclrtCreateStream");

  aclDataType dtype = useFP16 ? ACL_FLOAT16 : ACL_FLOAT;
  size_t eltSize = useFP16 ? sizeof(aclFloat16) : sizeof(float);

  // Allocate device buffers
  void* deviceInput = ascendMalloc(numInputFloats * eltSize);
  void* deviceMask = ascendMalloc(numMaskFloats * eltSize);
  float* deviceMaskSum = (float*)ascendMalloc(numMaskSumFloats * sizeof(float));
  void* deviceGpoolConvOut = ascendMalloc((size_t)batchSize * desc->gpoolConv.outChannels * nnXLen * nnYLen * eltSize);
  void* deviceMidBuf = ascendMalloc(numOutputFloats * eltSize);
  void* deviceTrunkScratch = ascendMalloc(numInputFloats * eltSize);

  // Compute gpool scratch size
  size_t gpoolElts = (size_t)batchSize * desc->gpoolConv.outChannels;
  size_t gpoolBytesDtype = gpoolElts * (useFP16 ? sizeof(aclFloat16) : sizeof(float));
  size_t gpoolBytesFP32 = gpoolElts * sizeof(float);
  size_t gpoolScratchBytes = 2 * gpoolBytesDtype + 5 * gpoolBytesFP32
                           + gpoolElts * 3 * sizeof(float)
                           + (size_t)batchSize * desc->regularConv.outChannels * sizeof(float);
  void* deviceGpoolScratch = ascendMalloc(gpoolScratchBytes);

  // Copy inputs to device
  ascendCopyH2D(deviceInput, inputBuffer.data(), numInputFloats * sizeof(float));
  ascendCopyH2D(deviceMask, maskBuffer.data(), numMaskFloats * sizeof(float));

  // Compute mask sum on host and copy to device (simplified for testing)
  vector<float> maskSumHost(batchSize, 0.0f);
  for(int n = 0; n < batchSize; n++) {
    for(int y = 0; y < nnYLen; y++) {
      for(int x = 0; x < nnXLen; x++) {
        int idx = n * nnXLen * nnYLen + y * nnXLen + x;
        maskSumHost[n] += maskBuffer[idx];
      }
    }
  }
  ACL_CHECK(aclrtMemcpy(deviceMaskSum, numMaskSumFloats * sizeof(float),
                        maskSumHost.data(), numMaskSumFloats * sizeof(float),
                        ACL_MEMCPY_HOST_TO_DEVICE), "aclrtMemcpy maskSum");

  // Create GlobalPoolingResidualBlock
  GlobalPoolingResidualBlock* residualBlock = new GlobalPoolingResidualBlock(desc, nnXLen, nnYLen, useFP16);

  // Get workspace size (takes only batchSize and stream)
  size_t workspaceBytes = residualBlock->requiredWorkspaceBytes(batchSize, stream);
  void* deviceWorkspace = nullptr;
  if(workspaceBytes > 0) {
    deviceWorkspace = ascendMalloc(workspaceBytes);
  }

  // Create temporary handle for tensorCache access
  ComputeHandle* tempHandle = new ComputeHandle(0, useFP16, nnXLen, nnYLen, false, false);
  tempHandle->stream = stream;
  tempHandle->initScalars();
  tempHandle->model = nullptr;
  tempHandle->buffers = nullptr;

  // Apply global pooling residual block
  residualBlock->apply(tempHandle, stream, batchSize, deviceMask, deviceMaskSum,
                       deviceInput, deviceTrunkScratch, deviceMidBuf,
                       deviceGpoolConvOut, deviceGpoolScratch,
                       deviceWorkspace, workspaceBytes);

  // Synchronize
  ACL_CHECK(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");

  // Copy output back to host (output is in deviceInput after residual add)
  outputBuffer.resize(numOutputFloats);
  ascendCopyD2H(outputBuffer.data(), deviceInput, numOutputFloats * sizeof(float));

  // Cleanup
  ascendFree(deviceWorkspace);
  ascendFree(deviceInput);
  ascendFree(deviceMask);
  ascendFree(deviceMaskSum);
  ascendFree(deviceGpoolConvOut);
  ascendFree(deviceMidBuf);
  ascendFree(deviceTrunkScratch);
  ascendFree(deviceGpoolScratch);
  delete residualBlock;
  // Clean up temp handle (stream is owned by us, don't destroy it)
  tempHandle->stream = nullptr;
  delete tempHandle;
  ACL_CHECK(aclrtDestroyStream(stream), "aclrtDestroyStream");

  return true;
}

#endif // USE_ASCEND_BACKEND
