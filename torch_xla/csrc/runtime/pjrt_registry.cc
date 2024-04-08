#include "torch_xla/csrc/runtime/pjrt_registry.h"

#include "absl/log/initialize.h"
#include "torch_xla/csrc/runtime/debug_macros.h"
#include "torch_xla/csrc/runtime/env_vars.h"
#include "torch_xla/csrc/runtime/profiler.h"
#include "torch_xla/csrc/runtime/sys_util.h"
#include "torch_xla/csrc/runtime/tf_logging.h"
#include "torch_xla/csrc/runtime/xla_coordinator.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/pjrt/distributed/client.h"
#include "xla/pjrt/distributed/distributed.h"
#include "xla/pjrt/distributed/in_memory_key_value_store.h"
#include "xla/pjrt/gpu/se_gpu_pjrt_client.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/tfrt_cpu_pjrt_client.h"

// piz extra include for pjrtcient
#include "xla/service/hlo_cost_analysis.h"
namespace xla {
namespace {


  PjRtCompileOnlyDevice::PjRtCompileOnlyDevice(const PjRtDeviceDescription* description)
      : description_(std::move(description)) {}

  const PjRtDeviceDescription& PjRtCompileOnlyDevice::description() const  {
    return *description_;
  }

  PjRtClient* PjRtCompileOnlyDevice::client() const  { return nullptr; }
  bool PjRtCompileOnlyDevice::IsAddressable() const  { return false; }
  int PjRtCompileOnlyDevice::local_hardware_id() const  {
    return local_hardware_id_typed().value();
  }

  PjRtLocalDeviceId PjRtCompileOnlyDevice::local_device_id() const  {
    return PjRtLocalDeviceId(local_hardware_id_typed().value());
  }

  PjRtLocalHardwareId PjRtCompileOnlyDevice::local_hardware_id_typed() const  {
    return PjRtLocalHardwareId(-1);
  }

  std::unique_ptr<ScopedAsyncTrackingEvent> PjRtCompileOnlyDevice::CreateAsyncTrackingEvent(
      absl::string_view description) const  {
    return nullptr;
  }
  Status PjRtCompileOnlyDevice::TransferToInfeed(const LiteralSlice& literal)  {
    return Unimplemented("TransferToInfeed is not supported");
  }
  Status PjRtCompileOnlyDevice::TransferFromOutfeed(MutableBorrowingLiteral literal)  {
    return Unimplemented("TransferFromOutfeed is not supported");
  }
  absl::Span<PjRtMemorySpace* const> PjRtCompileOnlyDevice::memory_spaces() const  {
    return {};
  }
  StatusOr<PjRtMemorySpace*> PjRtCompileOnlyDevice::default_memory_space() const  {
    return Unimplemented("default_memory_space is not supported");
  }



// class InvalidIfrtCompiler final
//     : public llvm::RTTIExtends<InvalidIfrtCompiler, ifrt::Compiler> {
//  public:
//   StatusOr<std::unique_ptr<ifrt::LoadedExecutable>> Compile(
//       std::unique_ptr<ifrt::Program> program,
//       std::unique_ptr<ifrt::CompileOptions> options) override {
//     return Unimplemented("Compile not implemented.");
//   }

//   StatusOr<std::unique_ptr<ifrt::LoadedExecutable>> DeserializeLoadedExecutable(
//       absl::string_view serialized,
//       std::unique_ptr<ifrt::DeserializeExecutableOptions> options) override {
//     return Unimplemented("DeserializeLoadedExecutable not implemented.");
//   }

//   static char ID;  // NOLINT
// };
// char InvalidIfrtCompiler::ID = 0;  // NOLINT

 CompileOnlyPjRtClient::CompileOnlyPjRtClient(std::shared_ptr<PjRtTopologyDescription> topology): topology_(std::move(topology)), descriptions_(topology_->DeviceDescriptions()) {
 for (auto& description : descriptions_) {
      owned_devices_.push_back(
          std::make_unique<PjRtCompileOnlyDevice>(description.get()));
      devices_.push_back(owned_devices_.back().get());
      // devices_.back()->client()->platform_name() :  devices_.back()->client() invalid memory
    }
  }
  // implement those pure virtual methods:
  int CompileOnlyPjRtClient::device_count() const { return devices().size(); }
  int CompileOnlyPjRtClient::addressable_device_count() const { return 0; }
  absl::Span<PjRtDevice* const> CompileOnlyPjRtClient::devices() const { return devices_; }
  // PIZ: if we don't implement this, this will raise error in GetDefaultDevice()
  absl::Span<PjRtDevice* const> CompileOnlyPjRtClient::addressable_devices() const {
    return {};
  }
  int CompileOnlyPjRtClient::process_index() const { return 0; }

  StatusOr<PjRtDevice*> CompileOnlyPjRtClient::LookupDevice(
        PjRtGlobalDeviceId global_device_id) const {
          return Unimplemented("LookupDevice not available with compile-only client.");
        }
  
   StatusOr<PjRtDevice*> CompileOnlyPjRtClient::LookupAddressableDevice(
        PjRtLocalDeviceId local_device_id) const {
      return Unimplemented(
          "LookupAddressableDevice not available with compile-only client.");
    } 
  
    absl::Span<PjRtMemorySpace* const> CompileOnlyPjRtClient::memory_spaces() const {
      return {};
    }

    PjRtRuntimeType CompileOnlyPjRtClient::runtime_type() const {
      return PjRtRuntimeType::kTfrt;
    }

    absl::string_view CompileOnlyPjRtClient::platform_name() const {
      return topology_->platform_name();
    }
    absl::string_view CompileOnlyPjRtClient::platform_version() const {
      return topology_->platform_version();
    }
    PjRtPlatformId CompileOnlyPjRtClient::platform_id() const {
      return topology_->platform_id();
    }
    StatusOr<DeviceAssignment> CompileOnlyPjRtClient::GetDefaultDeviceAssignment(
      int num_replicas, int num_partitions) const {
      return Unimplemented(
          "GetDefaultDeviceAssignment not available with compile-only client.");
    }
    StatusOr<Layout> CompileOnlyPjRtClient::GetDefaultLayout(PrimitiveType element_type,
                                            absl::Span<const int64_t> dims) {
      return Unimplemented(
          "GetDefaultLayout not available with compile-only client.");
    }
                                        
    StatusOr<std::unique_ptr<HloCostAnalysis>> CompileOnlyPjRtClient::GetHloCostAnalysis() const {
       return Unimplemented("");
    }

    StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileOnlyPjRtClient::Compile(
      const XlaComputation& computation, CompileOptions options) {
       return Unimplemented("");
    }

    // StatusOr<std::unique_ptr<PjRtExecutable>> CompileOnlyPjRtClient::CompileUnloaded(
    //   const XlaComputation& computation, CompileOptions options) {
    //     return PjRtCompile(options, computation, *topology_);
    // }

    StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileOnlyPjRtClient::Compile(
      mlir::ModuleOp module, CompileOptions options) {
       return Unimplemented("");

    }

    StatusOr<std::unique_ptr<PjRtLoadedExecutable>> CompileOnlyPjRtClient::DeserializeExecutable(
        absl::string_view serialized, std::optional<CompileOptions> options) {
      return Unimplemented("DeserializeExecutable not implemented.");
   }

    StatusOr<std::unique_ptr<PjRtBuffer>> CompileOnlyPjRtClient::CreateUninitializedBuffer(
        const Shape& shape, PjRtDevice* device) {
      return Unimplemented("CreateUninitializedBuffer not implemented.");
    }


    StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
    CompileOnlyPjRtClient::CreateBuffersForAsyncHostToDevice(absl::Span<const Shape> shapes,
                                    PjRtMemorySpace* memory_space) {
                                      return Unimplemented("");
                                    }


    StatusOr<std::unique_ptr<PjRtClient::AsyncHostToDeviceTransferManager>>
      CompileOnlyPjRtClient::CreateBuffersForAsyncHostToDevice(absl::Span<const Shape> shapes,
                                    PjRtDevice* device) {
                                      return Unimplemented("");
                                    }



    StatusOr<std::unique_ptr<PjRtBuffer>> CompileOnlyPjRtClient::BufferFromHostBuffer(
      const void* data, PrimitiveType type, absl::Span<int64_t const> dims,
      std::optional<absl::Span<int64_t const>> byte_strides,
      HostBufferSemantics host_buffer_semantics,
      absl::AnyInvocable<void() &&> on_done_with_host_buffer,
      PjRtDevice* device) {
        // return Unimplemented{""};
        return tsl::errors::Unimplemented(
        "BufferFromHostBuffer with an optional device layout is not "
        "implemented on platform: ",
        platform_name());
      }    




    StatusOr<std::unique_ptr<PjRtBuffer>> CompileOnlyPjRtClient::BufferFromHostLiteral(
      const LiteralSlice& literal, PjRtDevice* device) {
        return Unimplemented("");
      }


    StatusOr<std::unique_ptr<PjRtBuffer>> CompileOnlyPjRtClient::CreateViewOfDeviceBuffer(
      void* device_ptr, const Shape& shape, PjRtDevice* device,
      std::function<void()> on_delete_callback,
      std::optional<std::intptr_t> stream = std::nullopt) {
        return Unimplemented("");
      }

    StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
      CompileOnlyPjRtClient::MakeCrossHostReceiveBuffers(absl::Span<const Shape> shapes,
                              PjRtDevice* device,
                              PjRtCrossHostRecvNotifier notifier) {
                                return Unimplemented("");
                              }

    StatusOr<std::vector<std::unique_ptr<PjRtBuffer>>>
      CompileOnlyPjRtClient::MakeCrossHostReceiveBuffersForGather(
        absl::Span<const Shape> shapes, std::vector<GatherDetails> gather_details,
        PjRtDevice* device, PjRtCrossHostRecvNotifier notifier)  {
      return Unimplemented("");  
    }
    
    StatusOr<ChannelHandle> CompileOnlyPjRtClient::CreateChannelHandle() {
      return Unimplemented(""); 
    }

    StatusOr<ChannelHandle> CompileOnlyPjRtClient::CreateDeviceToHostChannelHandle() {
      return Unimplemented(""); 
    }

    StatusOr<ChannelHandle> CompileOnlyPjRtClient::CreateHostToDeviceChannelHandle() {
      return Unimplemented(""); 
    }
    
    Status CompileOnlyPjRtClient::Defragment() {
      return Unimplemented(""); 
    }


  // private:
  //   std::shared_ptr<PjRtTopologyDescription> topology_;
  //   //  InvalidIfrtCompiler default_compiler_;
  //   std::vector<std::unique_ptr<const PjRtDeviceDescription>> descriptions_;
  //   std::vector<std::unique_ptr<PjRtCompileOnlyDevice>> owned_devices_;
  //   std::vector<PjRtDevice*> devices_; 

}
}

namespace torch_xla {
namespace runtime {

namespace {

std::unordered_map<std::string, std::shared_ptr<const PjRtPlugin>>
    pjrt_plugins_;

xla::GpuAllocatorConfig GetGpuAllocatorConfig() {
  auto allocator_config = xla::GpuAllocatorConfig{};
  if (sys_util::GetEnvString(env::kEnvPjrtAllocatorCudaAsync, "").empty() &&
      sys_util::GetEnvString(env::kEnvPjrtAllocatorPreallocate, "").empty() &&
      sys_util::GetEnvString(env::kEnvPjrtAllocatorFraction, "").empty()) {
    return allocator_config;
  }
  if (sys_util::GetEnvBool(env::kEnvPjrtAllocatorCudaAsync, false)) {
    allocator_config.kind = xla::GpuAllocatorConfig::Kind::kCudaAsync;
  }
  allocator_config.preallocate =
      sys_util::GetEnvBool(env::kEnvPjrtAllocatorPreallocate, true);
  allocator_config.memory_fraction =
      sys_util::GetEnvDouble(env::kEnvPjrtAllocatorFraction, 0.75);
  return allocator_config;
}

std::shared_ptr<const PjRtPlugin> GetPjRtPlugin(
    const std::string& device_type) {
  auto plugin_path = pjrt_plugins_.find(device_type);
  return plugin_path != pjrt_plugins_.end() ? plugin_path->second : nullptr;
}

}  // namespace

void RegisterPjRtPlugin(std::string name,
                        std::shared_ptr<const PjRtPlugin> plugin) {
  TF_VLOG(3) << "Registering PjRt plugin " << name;
  pjrt_plugins_[name] = plugin;
}

std::tuple<std::unique_ptr<xla::PjRtClient>, std::unique_ptr<XlaCoordinator>>
InitializePjRt(const std::string& device_type) {
  std::cout << "device_type: " << device_type << std::endl;
  std::unique_ptr<xla::PjRtClient> client;
  std::unique_ptr<XlaCoordinator> coordinator;

  if (sys_util::GetEnvBool(env::kEnvPjrtDynamicPlugins, false)) {
    std::shared_ptr<const PjRtPlugin> plugin = GetPjRtPlugin(device_type);
    if (plugin) {
      TF_VLOG(1) << "Initializing client for PjRt plugin " << device_type;

      std::shared_ptr<xla::KeyValueStoreInterface> kv_store = nullptr;
      if (plugin->requires_xla_coordinator()) {
        int local_process_rank = sys_util::GetEnvInt(
            env::kEnvPjRtLocalRank, sys_util::GetEnvInt("LOCAL_RANK", 0));
        int global_process_rank =
            sys_util::GetEnvInt("RANK", local_process_rank);
        int local_world_size =
            sys_util::GetEnvInt(env::kEnvPjRtLocalProcessCount,
                                sys_util::GetEnvInt("LOCAL_WORLD_SIZE", 1));
        int global_world_size =
            sys_util::GetEnvInt("WORLD_SIZE", local_world_size);

        std::string master_addr =
            runtime::sys_util::GetEnvString("MASTER_ADDR", "localhost");
        std::string port = runtime::sys_util::GetEnvString(
            "XLA_COORDINATOR_PORT", XlaCoordinator::kDefaultCoordinatorPort);

        TF_VLOG(3) << "Creating coordinator for rank=" << global_process_rank
                   << ", world size=" << global_world_size
                   << ", coordinator address=" << master_addr << ":" << port;

        // Use the XlaCoordinator as the distributed key-value store.
        coordinator = std::make_unique<XlaCoordinator>(
            global_process_rank, global_world_size, master_addr, port);
        std::shared_ptr<xla::DistributedRuntimeClient> distributed_client =
            coordinator->GetClient();
        kv_store = xla::GetDistributedKeyValueStore(distributed_client,
                                                    /*key_prefix=*/"pjrt:");
      }
      const PJRT_Api* c_api = *pjrt::LoadPjrtPlugin(
          absl::AsciiStrToLower(device_type), plugin->library_path());
      XLA_CHECK_OK(pjrt::InitializePjrtPlugin(device_type));
      auto create_options = plugin->client_create_options();
      client = xla::GetCApiClient(
                   absl::AsciiStrToUpper(device_type),
                   {create_options.begin(), create_options.end()}, kv_store)
                   .value();
      profiler::RegisterProfilerForPlugin(c_api);
    }
  } else if (device_type == "CPU") {
    TF_VLOG(1) << "Initializing PjRt CPU client...";
    bool async = sys_util::GetEnvBool(env::kEnvPjrtAsyncCpuClient, true);
    int cpu_device_count = sys_util::GetEnvInt(env::kEnvNumCpu, 1);
    client = std::move(xla::GetTfrtCpuClient(async, cpu_device_count).value());
  } else if (device_type == "TPU") {
    TF_VLOG(1) << "Initializing TFRT TPU client...";
    // Init the absl logging to avoid the log spam.
    absl::InitializeLog();
    // Prefer $TPU_LIBRARY_PATH if set
    auto tpu_library_path = sys_util::GetEnvString(
        env::kEnvTpuLibraryPath,
        sys_util::GetEnvString(env::kEnvInferredTpuLibraryPath, "libtpu.so"));
    XLA_CHECK_OK(pjrt::LoadPjrtPlugin("tpu", tpu_library_path).status());
    xla::Status tpu_status = pjrt::InitializePjrtPlugin("tpu");
    XLA_CHECK_OK(tpu_status);
    client = std::move(xla::GetCApiClient("TPU").value());
    const PJRT_Api* c_api =
        static_cast<xla::PjRtCApiClient*>(client.get())->pjrt_c_api();
    profiler::RegisterProfilerForPlugin(c_api);
  } else if (device_type == "AOT") {
    TF_VLOG(1) << "Initializing AOT client...";
    // Prefer $TPU_LIBRARY_PATH if set
    auto tpu_library_path = sys_util::GetEnvString(
        env::kEnvTpuLibraryPath,
        sys_util::GetEnvString(env::kEnvInferredTpuLibraryPath,
        "libtpu.so"));
    XLA_CHECK_OK(pjrt::LoadPjrtPlugin("tpu", tpu_library_path).status());
    tsl::Status tpu_status = pjrt::InitializePjrtPlugin(
        "tpu");  // PIZ: issue here for fake tpu due to LIBTPU_INIT_ARGS
    XLA_CHECK_OK(tpu_status);
    // xla::PjRtTopologyDescription topo =
    std::string topology_name = "";
    absl::flat_hash_map<std::string, xla::PjRtValueType> create_options = {};
    absl::StatusOr<std::unique_ptr<xla::PjRtTopologyDescription>> topo = xla::GetCApiTopology("tpu", topology_name, create_options);
    XLA_CHECK_OK(topo.status()); 
    std::shared_ptr<xla::PjRtTopologyDescription> shared_topo = std::move(topo.value());
    client = std::move(
      std::make_unique<xla::CompileOnlyPjRtClient>(shared_topo)
    );
    std::cout << "piz111" << std::endl;
    // client = std::move(xla::GetCApiClient("TPU").value());  // PIZ: issue
    // here
  } else if (device_type == "TPU_LEGACY") {
    XLA_ERROR() << "TPU_LEGACY client is no longer available.";
  } else if (device_type == "CUDA") {
    TF_VLOG(1) << "Initializing PjRt GPU client...";
    bool async = sys_util::GetEnvBool(env::kEnvPjrtAsyncGpuClient, true);
    int local_process_rank = sys_util::GetEnvInt(env::kEnvPjRtLocalRank, 0);
    int global_process_rank = sys_util::GetEnvInt("RANK", local_process_rank);
    int local_world_size = sys_util::GetEnvInt("LOCAL_WORLD_SIZE", 1);
    int global_world_size = sys_util::GetEnvInt("WORLD_SIZE", local_world_size);

    TF_VLOG(3) << "Getting StreamExecutorGpuClient for node_id="
               << global_process_rank << ", num_nodes=" << global_world_size
               << ", local_process_rank=" << local_process_rank
               << ", local_world_size=" << local_world_size
               << ", spmd case=" << sys_util::GetEnvBool("XLA_USE_SPMD", false)
               << ", PJRT_LOCAL_PROCESS_RANK="
               << sys_util::GetEnvString(env::kEnvPjRtLocalRank, "")
               << ", RANK=" << sys_util::GetEnvString("RANK", "")
               << ", LOCAL_WORLD_SIZE="
               << sys_util::GetEnvString("LOCAL_WORLD_SIZE", "")
               << ", WORLD_SIZE=" << sys_util::GetEnvString("WORLD_SIZE", "");
    std::optional<std::set<int>> allowed_devices;
    if (local_world_size > 1) {
      allowed_devices = std::set{local_process_rank};
    }

    std::shared_ptr<xla::KeyValueStoreInterface> kv_store;
    if (global_world_size > 1) {
      // Use the distributed key-value store from DistributedRuntimeClient.
      std::string master_addr =
          runtime::sys_util::GetEnvString("MASTER_ADDR", "localhost");
      std::string port = runtime::sys_util::GetEnvString(
          "XLA_COORDINATOR_PORT", XlaCoordinator::kDefaultCoordinatorPort);
      coordinator = std::make_unique<XlaCoordinator>(
          global_process_rank, global_world_size, master_addr, port);
      std::shared_ptr<xla::DistributedRuntimeClient> distributed_client =
          coordinator->GetClient();
      kv_store = xla::GetDistributedKeyValueStore(distributed_client,
                                                  /*key_prefix=*/"gpu:");
    }

    xla::GpuClientOptions options;
    options.allocator_config = GetGpuAllocatorConfig();
    options.node_id = global_process_rank;
    options.num_nodes = global_world_size;
    options.allowed_devices = allowed_devices;
    options.platform_name = "gpu";
    options.should_stage_host_to_device_transfers = true;
    options.kv_store = kv_store;
    client = std::move(xla::GetStreamExecutorGpuClient(options).value());
  } else if (device_type == "XPU") {
    TF_VLOG(1) << "Initializing PjRt XPU client...";
    XLA_CHECK_OK(
        pjrt::LoadPjrtPlugin(
            "xpu", sys_util::GetEnvString(env::kEnvXpuLibraryPath, "libxpu.so"))
            .status());
    client = std::move(xla::GetCApiClient("XPU").value());
  } else if (device_type == "NEURON") {
    TF_VLOG(1) << "Initializing PjRt NEURON client...";
    XLA_CHECK_OK(pjrt::LoadPjrtPlugin("NEURON", sys_util::GetEnvString(
                                                    env::kEnvNeuronLibraryPath,
                                                    "libneuronpjrt.so"))
                     .status());
    client = std::move(xla::GetCApiClient("NEURON").value());
  }

  XLA_CHECK(client) << absl::StrFormat("Unknown %s '%s'", env::kEnvPjRtDevice,
                                       device_type);

  return {std::move(client), std::move(coordinator)};
}

}  // namespace runtime
}  // namespace torch_xla
