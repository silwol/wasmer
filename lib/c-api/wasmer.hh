// The Wasmer C/C++ header file.

#if !defined(WASMER_H_MACROS)

#define WASMER_H_MACROS

// Define the `ARCH_X86_X64` constant.
#if defined(MSVC) && defined(_M_AMD64)
#  define ARCH_X86_64
#elif (defined(GCC) || defined(__GNUC__) || defined(__clang__)) && defined(__x86_64__)
#  define ARCH_X86_64
#endif

// Compatibility with non-Clang compilers.
#if !defined(__has_attribute)
#  define __has_attribute(x) 0
#endif

// Compatibility with non-Clang compilers.
#if !defined(__has_declspec_attribute)
#  define __has_declspec_attribute(x) 0
#endif

// Define the `DEPRECATED` macro.
#if defined(GCC) || defined(__GNUC__) || __has_attribute(deprecated)
#  define DEPRECATED(message) __attribute__((deprecated(message)))
#elif defined(MSVC) || __has_declspec_attribute(deprecated)
#  define DEPRECATED(message) __declspec(deprecated(message))
#endif

// The `wasi` feature has been enabled for this build.
#define WASMER_WASI_ENABLED
#endif // WASMER_H_MACROS


//
// OK, here we go. The code below is automatically generated.
//


#ifndef WASMER_H
#define WASMER_H

#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <new>

#if defined(WASMER_WASI_ENABLED)
enum class Version : uint8_t {
#if defined(WASMER_WASI_ENABLED)
  /// Version cannot be detected or is unknown.
  Unknown = 0,
#endif
#if defined(WASMER_WASI_ENABLED)
  /// Latest version. See `wasmer_wasi::WasiVersion::Latest` to
  /// learn more.
  Latest = 1,
#endif
#if defined(WASMER_WASI_ENABLED)
  /// `wasi_unstable`.
  Snapshot0 = 2,
#endif
#if defined(WASMER_WASI_ENABLED)
  /// `wasi_snapshot_preview1`.
  Snapshot1 = 3,
#endif
};
#endif

enum class wasm_valkind_enum : uint8_t {
  WASM_I32 = 0,
  WASM_I64 = 1,
  WASM_F32 = 2,
  WASM_F64 = 3,
  WASM_ANYREF = 128,
  WASM_FUNCREF = 129,
};

/// List of export/import kinds.
enum class wasmer_import_export_kind : uint32_t {
  /// The export/import is a function.
  WASM_FUNCTION = 0,
  /// The export/import is a global.
  WASM_GLOBAL = 1,
  /// The export/import is a memory.
  WASM_MEMORY = 2,
  /// The export/import is a table.
  WASM_TABLE = 3,
};

/// The `wasmer_result_t` enum is a type that represents either a
/// success, or a failure.
enum class wasmer_result_t {
  /// Represents a success.
  WASMER_OK = 1,
  /// Represents a failure.
  WASMER_ERROR = 2,
};

/// Represents all possibles WebAssembly value types.
///
/// See `wasmer_value_t` to get a complete example.
enum class wasmer_value_tag : uint32_t {
  /// Represents the `i32` WebAssembly type.
  WASM_I32,
  /// Represents the `i64` WebAssembly type.
  WASM_I64,
  /// Represents the `f32` WebAssembly type.
  WASM_F32,
  /// Represents the `f64` WebAssembly type.
  WASM_F64,
};

template<typename T = void>
struct Arc;

template<typename T = void>
struct Box;

template<typename T = void>
struct Option;

#if defined(WASMER_WASI_ENABLED)
struct wasi_version_t;
#endif

struct wasm_engine_t;

struct wasm_ref_t;

#if defined(WASMER_WASI_ENABLED)
struct wasi_config_t {
  bool inherit_stdout;
  bool inherit_stderr;
  bool inherit_stdin;
  WasiStateBuilder state_builder;
};
#endif

#if defined(WASMER_WASI_ENABLED)
struct wasi_env_t {
  WasiEnv inner;
};
#endif

struct wasm_instance_t {
  Arc<Instance> inner;
};

struct wasm_memory_t {
  Memory inner;
};

/// Opaque wrapper around `Store`
struct wasm_store_t {

};

struct wasm_module_t {
  Arc<Module> inner;
};

struct wasm_extern_t {
  Option<Arc<Instance>> instance;
  Extern inner;
};

/// this can be a wasmer-specific type with wasmer-specific functions for manipulating it
struct wasm_config_t {

};

using wasm_name_t = wasm_byte_vec_t;

using wasm_externtype_t = ExternType;

struct wasm_exporttype_t {
  wasm_name_t *name;
  wasm_externtype_t *extern_type;
};

struct wasm_func_t {
  Function inner;
  Option<Arc<Instance>> instance;
};

struct wasm_global_t {
  Global inner;
};

struct wasm_table_t {
  Table inner;
};

using wasm_externkind_t = uint8_t;

using wasm_functype_t = wasm_externtype_t;

using wasm_globaltype_t = wasm_externtype_t;

using wasm_memorytype_t = wasm_externtype_t;

struct wasm_tabletype_t {
  wasm_externtype_t extern_;
};

struct wasm_trap_t {

};

using wasm_valkind_t = uint8_t;

union wasm_val_inner {
  int32_t int32_t;
  int64_t int64_t;
  float float32_t;
  double float64_t;
  wasm_ref_t *wref;
};

struct wasm_val_t {
  wasm_valkind_t kind;
  wasm_val_inner of;
};

using wasm_func_callback_t = wasm_trap_t*(*)(const wasm_val_t *args, wasm_val_t *results);

using wasm_func_callback_with_env_t = wasm_trap_t*(*)(void*, const wasm_val_t *args, wasm_val_t *results);

using wasm_env_finalizer_t = void(*)(void);

struct wasm_valtype_t {
  wasm_valkind_enum valkind;
};

using wasm_mutability_t = uint8_t;

struct wasm_importtype_t {
  wasm_name_t *module;
  wasm_name_t *name;
  wasm_externtype_t *extern_type;
};

struct wasm_limits_t {
  uint32_t min;
  uint32_t max;
};

using wasm_table_size_t = uint32_t;

struct wasmer_module_t {

};

/// Opaque pointer to an Instance type plus metadata.
///
/// This type represents a WebAssembly instance. It
/// is generally generated by the `wasmer_instantiate()` function, or by
/// the `wasmer_module_instantiate()` function for the most common paths.
struct wasmer_instance_t {

};

struct wasmer_byte_array {
  const uint8_t *bytes;
  uint32_t bytes_len;
};

#if defined(WASMER_EMSCRIPTEN_ENABLED)
/// Type used to construct an import_object_t with Emscripten imports.
struct wasmer_emscripten_globals_t {

};
#endif

struct wasmer_import_object_t {

};

/// Opaque pointer to `NamedExportDescriptor`.
struct wasmer_export_descriptor_t {

};

/// Opaque pointer to `NamedExportDescriptors`.
struct wasmer_export_descriptors_t {

};

/// Opaque pointer to `wasmer_export_t`.
struct wasmer_export_func_t {

};

/// Represents a WebAssembly value.
///
/// This is a [Rust union][rust-union], which is equivalent to the C
/// union. See `wasmer_value_t` to get a complete example.
///
/// [rust-union]: https://doc.rust-lang.org/reference/items/unions.html
union wasmer_value {
  int32_t I32;
  int64_t I64;
  float F32;
  double F64;
};

/// Represents a WebAssembly type and value pair,
/// i.e. `wasmer_value_tag` and `wasmer_value`. Since the latter is an
/// union, it's the safe way to read or write a WebAssembly value in
/// C.
///
/// Example:
///
/// ```c
/// // Create a WebAssembly value.
/// wasmer_value_t wasm_value = {
///     .tag = WASM_I32,
///     .value.I32 = 42,
/// };
///
/// // Read a WebAssembly value.
/// if (wasm_value.tag == WASM_I32) {
///     int32_t x = wasm_value.value.I32;
///     // …
/// }
/// ```
struct wasmer_value_t {
  /// The value type.
  wasmer_value_tag tag;
  /// The value.
  wasmer_value value;
};

/// Opaque pointer to `ImportType`.
struct wasmer_export_t {

};

/// Opaque pointer to a `wasmer_vm::Memory` value in Rust.
///
/// A `wasmer_vm::Memory` represents a WebAssembly memory. It is
/// possible to create one with `wasmer_memory_new()` and pass it as
/// imports of an instance, or to read it from exports of an instance
/// with `wasmer_export_to_memory()`.
struct wasmer_memory_t {

};

/// Opaque pointer to the opaque structure
/// `crate::deprecated::NamedExports`, which is a wrapper around a
/// vector of the opaque structure `crate::deprecated::NamedExport`.
///
/// Check the `wasmer_instance_exports()` function to learn more.
struct wasmer_exports_t {

};

struct wasmer_global_t {

};

struct wasmer_global_descriptor_t {
  bool mutable_;
  wasmer_value_tag kind;
};

struct wasmer_import_descriptor_t {

};

struct wasmer_import_descriptors_t {

};

struct wasmer_import_func_t {

};

struct wasmer_table_t {

};

/// Union of import/export value.
union wasmer_import_export_value {
  const wasmer_import_func_t *func;
  const wasmer_table_t *table;
  const wasmer_memory_t *memory;
  const wasmer_global_t *global;
};

struct wasmer_import_t {
  wasmer_byte_array module_name;
  wasmer_byte_array import_name;
  wasmer_import_export_kind tag;
  wasmer_import_export_value value;
};

struct wasmer_import_object_iter_t {

};

/// Opaque pointer to a `wasmer_vm::Ctx` value in Rust.
///
/// An instance context is passed to any host function (aka imported
/// function) as the first argument. It is necessary to read the
/// instance data or the memory, respectively with the
/// `wasmer_instance_context_data_get()` function, and the
/// `wasmer_instance_context_memory()` function.
///
/// It is also possible to get the instance context outside a host
/// function by using the `wasmer_instance_context_get()`
/// function. See also `wasmer_instance_context_data_set()` to set the
/// instance context data.
///
/// Example:
///
/// ```c
/// // A host function that prints data from the WebAssembly memory to
/// // the standard output.
/// void print(wasmer_instance_context_t *context, int32_t pointer, int32_t length) {
///     // Use `wasmer_instance_context` to get back the first instance memory.
///     const wasmer_memory_t *memory = wasmer_instance_context_memory(context, 0);
///
///     // Continue…
/// }
/// ```
struct wasmer_instance_context_t {

};

/// The `wasmer_limit_option_t` struct represents an optional limit
/// for `wasmer_limits_t`.
struct wasmer_limit_option_t {
  /// Whether the limit is set.
  bool has_some;
  /// The limit value.
  uint32_t some;
};

/// The `wasmer_limits_t` struct is a type that describes the limits of something
/// such as a memory or a table. See the `wasmer_memory_new()` function to get
/// more information.
struct wasmer_limits_t {
  /// The minimum number of allowed pages.
  uint32_t min;
  /// The maximum number of allowed pages.
  wasmer_limit_option_t max;
};

struct wasmer_serialized_module_t {

};

#if defined(WASMER_WASI_ENABLED)
/// Opens a directory that's visible to the WASI module as `alias` but
/// is backed by the host file at `host_file_path`
struct wasmer_wasi_map_dir_entry_t {
  /// What the WASI module will see in its virtual root
  wasmer_byte_array alias;
  /// The backing file that the WASI module will interact with via the alias
  wasmer_byte_array host_file_path;
};
#endif

extern "C" {

#if defined(WASMER_WASI_ENABLED)
void wasi_config_arg(wasi_config_t *config, const char *arg);
#endif

#if defined(WASMER_WASI_ENABLED)
void wasi_config_env(wasi_config_t *config, const char *key, const char *value);
#endif

#if defined(WASMER_WASI_ENABLED)
void wasi_config_inherit_stderr(wasi_config_t *config);
#endif

#if defined(WASMER_WASI_ENABLED)
void wasi_config_inherit_stdin(wasi_config_t *config);
#endif

#if defined(WASMER_WASI_ENABLED)
void wasi_config_inherit_stdout(wasi_config_t *config);
#endif

#if defined(WASMER_WASI_ENABLED)
Option<Box<wasi_config_t>> wasi_config_new(const char *program_name);
#endif

#if defined(WASMER_WASI_ENABLED)
void wasi_env_delete(Option<Box<wasi_env_t>> _state);
#endif

#if defined(WASMER_WASI_ENABLED)
/// Takes ownership over the `wasi_config_t`.
Option<Box<wasi_env_t>> wasi_env_new(Box<wasi_config_t> config);
#endif

#if defined(WASMER_WASI_ENABLED)
intptr_t wasi_env_read_stderr(wasi_env_t *env, char *buffer, uintptr_t buffer_len);
#endif

#if defined(WASMER_WASI_ENABLED)
intptr_t wasi_env_read_stdout(wasi_env_t *env, char *buffer, uintptr_t buffer_len);
#endif

#if defined(WASMER_WASI_ENABLED)
bool wasi_env_set_instance(wasi_env_t *env, const wasm_instance_t *instance);
#endif

#if defined(WASMER_WASI_ENABLED)
void wasi_env_set_memory(wasi_env_t *env, const wasm_memory_t *memory);
#endif

#if defined(WASMER_WASI_ENABLED)
/// Takes ownership of `wasi_env_t`.
bool wasi_get_imports(wasm_store_t *store,
                      const wasm_module_t *module,
                      const wasi_env_t *wasi_env,
                      wasm_extern_t **imports);
#endif

#if defined(WASMER_WASI_ENABLED)
wasi_version_t wasi_get_wasi_version(const wasm_module_t *module);
#endif

wasm_config_t *wasm_config_new();

void wasm_engine_delete(Option<Box<wasm_engine_t>> _wasm_engine_address);

Box<wasm_engine_t> wasm_engine_new_with_config(wasm_config_t *_config_ptr);

const wasm_name_t *wasm_exporttype_name(const wasm_exporttype_t *et);

Box<wasm_exporttype_t> wasm_exporttype_new(wasm_name_t *name, wasm_externtype_t *extern_type);

const wasm_externtype_t *wasm_exporttype_type(const wasm_exporttype_t *et);

Option<Box<wasm_func_t>> wasm_extern_as_func(wasm_extern_t *extern_ptr);

Option<Box<wasm_global_t>> wasm_extern_as_global(wasm_extern_t *extern_ptr);

Option<Box<wasm_memory_t>> wasm_extern_as_memory(wasm_extern_t *extern_ptr);

Option<Box<wasm_table_t>> wasm_extern_as_table(wasm_extern_t *extern_ptr);

wasm_externkind_t wasm_extern_kind(const wasm_extern_t *e);

Box<wasm_externtype_t> wasm_extern_type(const wasm_extern_t *e);

const wasm_functype_t *wasm_externtype_as_functype(const wasm_externtype_t *et);

const wasm_functype_t *wasm_externtype_as_functype_const(const wasm_externtype_t *et);

const wasm_globaltype_t *wasm_externtype_as_globaltype(const wasm_externtype_t *et);

const wasm_globaltype_t *wasm_externtype_as_globaltype_const(const wasm_externtype_t *et);

const wasm_memorytype_t *wasm_externtype_as_memorytype(const wasm_externtype_t *et);

const wasm_memorytype_t *wasm_externtype_as_memorytype_const(const wasm_externtype_t *et);

const wasm_tabletype_t *wasm_externtype_as_tabletype(const wasm_externtype_t *et);

const wasm_tabletype_t *wasm_externtype_as_tabletype_const(const wasm_externtype_t *et);

void wasm_externtype_delete(Option<Box<wasm_externtype_t>> _et);

wasm_externkind_t wasm_externtype_kind(const wasm_externtype_t *et);

Option<Box<wasm_extern_t>> wasm_func_as_extern(wasm_func_t *func_ptr);

wasm_trap_t *wasm_func_call(const wasm_func_t *func, const wasm_val_t *args, wasm_val_t *results);

void wasm_func_delete(Option<Box<wasm_func_t>> _func);

Option<Box<wasm_func_t>> wasm_func_new(wasm_store_t *store,
                                       const wasm_functype_t *ft,
                                       wasm_func_callback_t callback);

Option<Box<wasm_func_t>> wasm_func_new_with_env(wasm_store_t *store,
                                                const wasm_functype_t *ft,
                                                wasm_func_callback_with_env_t callback,
                                                void *env,
                                                wasm_env_finalizer_t finalizer);

uintptr_t wasm_func_param_arity(const wasm_func_t *func);

uintptr_t wasm_func_result_arity(const wasm_func_t *func);

const wasm_externtype_t *wasm_functype_as_externtype(const wasm_functype_t *ft);

const wasm_externtype_t *wasm_functype_as_externtype_const(const wasm_functype_t *ft);

Option<Box<wasm_functype_t>> wasm_functype_copy(wasm_functype_t *arg);

void wasm_functype_delete(Option<Box<wasm_functype_t>> _ft);

Option<Box<wasm_functype_t>> wasm_functype_new(wasm_valtype_vec_t *params,
                                               wasm_valtype_vec_t *results);

const wasm_valtype_vec_t *wasm_functype_params(const wasm_functype_t *ft);

const wasm_valtype_vec_t *wasm_functype_results(const wasm_functype_t *ft);

Option<Box<wasm_extern_t>> wasm_global_as_extern(wasm_global_t *global_ptr);

Box<wasm_global_t> wasm_global_copy(const wasm_global_t *wasm_global);

void wasm_global_delete(Option<Box<wasm_global_t>> _global);

void wasm_global_get(const wasm_global_t *wasm_global, wasm_val_t *out);

Option<Box<wasm_global_t>> wasm_global_new(wasm_store_t *store_ptr,
                                           const wasm_globaltype_t *gt,
                                           const wasm_val_t *val);

bool wasm_global_same(const wasm_global_t *wasm_global1, const wasm_global_t *wasm_global2);

void wasm_global_set(wasm_global_t *wasm_global, const wasm_val_t *val);

const wasm_externtype_t *wasm_globaltype_as_externtype(const wasm_globaltype_t *gt);

const wasm_externtype_t *wasm_globaltype_as_externtype_const(const wasm_globaltype_t *gt);

const wasm_valtype_t *wasm_globaltype_content(const wasm_globaltype_t *globaltype);

void wasm_globaltype_delete(Option<Box<wasm_globaltype_t>> _globaltype);

wasm_mutability_t wasm_globaltype_mutability(const wasm_globaltype_t *globaltype);

Option<Box<wasm_globaltype_t>> wasm_globaltype_new(Option<Box<wasm_valtype_t>> valtype,
                                                   wasm_mutability_t mutability);

void wasm_importtype_delete(Option<Box<wasm_importtype_t>> _importtype);

const wasm_name_t *wasm_importtype_module(const wasm_importtype_t *et);

const wasm_name_t *wasm_importtype_name(const wasm_importtype_t *et);

Box<wasm_importtype_t> wasm_importtype_new(wasm_name_t *module,
                                           wasm_name_t *name,
                                           wasm_externtype_t *extern_type);

const wasm_externtype_t *wasm_importtype_type(const wasm_importtype_t *et);

void wasm_instance_delete(Option<Box<wasm_instance_t>> _instance);

void wasm_instance_exports(const wasm_instance_t *instance, wasm_extern_vec_t *out);

Option<Box<wasm_instance_t>> wasm_instance_new(wasm_store_t *store,
                                               const wasm_module_t *module,
                                               const wasm_extern_t *const *imports,
                                               wasm_trap_t **_traps);

Option<Box<wasm_extern_t>> wasm_memory_as_extern(wasm_memory_t *memory_ptr);

Box<wasm_memory_t> wasm_memory_copy(const wasm_memory_t *wasm_memory);

uint8_t *wasm_memory_data(wasm_memory_t *memory);

uintptr_t wasm_memory_data_size(const wasm_memory_t *memory);

void wasm_memory_delete(Option<Box<wasm_memory_t>> _memory);

bool wasm_memory_grow(wasm_memory_t *memory, uint32_t delta);

Option<Box<wasm_memory_t>> wasm_memory_new(wasm_store_t *store_ptr, const wasm_memorytype_t *mt);

bool wasm_memory_same(const wasm_memory_t *wasm_memory1, const wasm_memory_t *wasm_memory2);

uint32_t wasm_memory_size(const wasm_memory_t *memory);

wasm_memorytype_t *wasm_memory_type(const wasm_memory_t *_memory_ptr);

const wasm_externtype_t *wasm_memorytype_as_externtype(const wasm_memorytype_t *mt);

const wasm_externtype_t *wasm_memorytype_as_externtype_const(const wasm_memorytype_t *mt);

void wasm_memorytype_delete(Option<Box<wasm_memorytype_t>> _memorytype);

const wasm_limits_t *wasm_memorytype_limits(const wasm_memorytype_t *mt);

Box<wasm_memorytype_t> wasm_memorytype_new(const wasm_limits_t *limits);

void wasm_module_delete(Option<Box<wasm_module_t>> _module);

wasm_module_t *wasm_module_deserialize(wasm_store_t *store_ptr, const wasm_byte_vec_t *bytes);

void wasm_module_exports(const wasm_module_t *module, wasm_exporttype_vec_t *out);

void wasm_module_imports(const wasm_module_t *module, wasm_importtype_vec_t *out);

Option<Box<wasm_module_t>> wasm_module_new(wasm_store_t *store_ptr, const wasm_byte_vec_t *bytes);

void wasm_module_serialize(const wasm_module_t *module, wasm_byte_vec_t *out_ptr);

void wasm_store_delete(wasm_store_t *wasm_store);

wasm_store_t *wasm_store_new(wasm_engine_t *wasm_engine_ptr);

Option<Box<wasm_extern_t>> wasm_table_as_extern(wasm_table_t *table_ptr);

Box<wasm_table_t> wasm_table_copy(const wasm_table_t *wasm_table);

void wasm_table_delete(Option<Box<wasm_table_t>> _table);

bool wasm_table_grow(wasm_table_t *_wasm_table, wasm_table_size_t _delta, wasm_ref_t *_init);

Option<Box<wasm_table_t>> wasm_table_new(wasm_store_t *store_ptr,
                                         const wasm_tabletype_t *tt,
                                         const wasm_ref_t *init);

bool wasm_table_same(const wasm_table_t *wasm_table1, const wasm_table_t *wasm_table2);

uintptr_t wasm_table_size(const wasm_table_t *wasm_table);

const wasm_externtype_t *wasm_tabletype_as_externtype(const wasm_tabletype_t *tt);

const wasm_externtype_t *wasm_tabletype_as_externtype_const(const wasm_tabletype_t *tt);

void wasm_tabletype_delete(Option<Box<wasm_tabletype_t>> _tabletype);

const wasm_valtype_t *wasm_tabletype_element(const wasm_tabletype_t *tabletype);

const wasm_limits_t *wasm_tabletype_limits(const wasm_tabletype_t *tabletype);

Box<wasm_tabletype_t> wasm_tabletype_new(Box<wasm_valtype_t> valtype, const wasm_limits_t *limits);

void wasm_trap_delete(wasm_trap_t *trap);

void wasm_trap_message(const wasm_trap_t *trap, wasm_byte_vec_t *out_ptr);

void wasm_val_copy(wasm_val_t *out_ptr, const wasm_val_t *val);

void wasm_val_delete(wasm_val_t *val);

void wasm_valtype_delete(Option<Box<wasm_valtype_t>> _valtype);

wasm_valkind_t wasm_valtype_kind(const wasm_valtype_t *valtype);

Option<Box<wasm_valtype_t>> wasm_valtype_new(wasm_valkind_t kind);

/// Creates a new Module from the given wasm bytes.
///
/// Returns `wasmer_result_t::WASMER_OK` upon success.
///
/// Returns `wasmer_result_t::WASMER_ERROR` upon failure. Use `wasmer_last_error_length`
/// and `wasmer_last_error_message` to get an error message.
wasmer_result_t wasmer_compile(wasmer_module_t **module,
                               uint8_t *wasm_bytes,
                               uint32_t wasm_bytes_len);

#if defined(WASMER_EMSCRIPTEN_ENABLED)
/// Convenience function for setting up arguments and calling the Emscripten
/// main function.
///
/// WARNING:
///
/// Do not call this function on untrusted code when operating without
/// additional sandboxing in place.
/// Emscripten has access to many host system calls and therefore may do very
/// bad things.
wasmer_result_t wasmer_emscripten_call_main(wasmer_instance_t *instance,
                                            const wasmer_byte_array *args,
                                            unsigned int args_len);
#endif

#if defined(WASMER_EMSCRIPTEN_ENABLED)
/// Destroy `wasmer_emscrpten_globals_t` created by
/// `wasmer_emscripten_get_emscripten_globals`.
void wasmer_emscripten_destroy_globals(wasmer_emscripten_globals_t *globals);
#endif

#if defined(WASMER_EMSCRIPTEN_ENABLED)
/// Create a `wasmer_import_object_t` with Emscripten imports, use
/// `wasmer_emscripten_get_emscripten_globals` to get a
/// `wasmer_emscripten_globals_t` from a `wasmer_module_t`.
///
/// WARNING:
///
/// This `import_object_t` contains thin-wrappers around host system calls.
/// Do not use this to execute untrusted code without additional sandboxing.
wasmer_import_object_t *wasmer_emscripten_generate_import_object(wasmer_emscripten_globals_t *globals);
#endif

#if defined(WASMER_EMSCRIPTEN_ENABLED)
/// Create a `wasmer_emscripten_globals_t` from a Wasm module.
wasmer_emscripten_globals_t *wasmer_emscripten_get_globals(const wasmer_module_t *module);
#endif

#if defined(WASMER_EMSCRIPTEN_ENABLED)
/// Execute global constructors (required if the module is compiled from C++)
/// and sets up the internal environment.
///
/// This function sets the data pointer in the same way that
/// [`wasmer_instance_context_data_set`] does.
wasmer_result_t wasmer_emscripten_set_up(wasmer_instance_t *instance,
                                         wasmer_emscripten_globals_t *globals);
#endif

/// Gets export descriptor kind
wasmer_import_export_kind wasmer_export_descriptor_kind(wasmer_export_descriptor_t *export_);

/// Gets name for the export descriptor
wasmer_byte_array wasmer_export_descriptor_name(wasmer_export_descriptor_t *export_descriptor);

/// Gets export descriptors for the given module
///
/// The caller owns the object and should call `wasmer_export_descriptors_destroy` to free it.
void wasmer_export_descriptors(const wasmer_module_t *module,
                               wasmer_export_descriptors_t **export_descriptors);

/// Frees the memory for the given export descriptors
void wasmer_export_descriptors_destroy(wasmer_export_descriptors_t *export_descriptors);

/// Gets export descriptor by index
wasmer_export_descriptor_t *wasmer_export_descriptors_get(wasmer_export_descriptors_t *export_descriptors,
                                                          int idx);

/// Gets the length of the export descriptors
int wasmer_export_descriptors_len(wasmer_export_descriptors_t *exports);

/// Calls a `func` with the provided parameters.
/// Results are set using the provided `results` pointer.
///
/// Returns `wasmer_result_t::WASMER_OK` upon success.
///
/// Returns `wasmer_result_t::WASMER_ERROR` upon failure. Use `wasmer_last_error_length`
/// and `wasmer_last_error_message` to get an error message.
wasmer_result_t wasmer_export_func_call(const wasmer_export_func_t *func,
                                        const wasmer_value_t *params,
                                        unsigned int params_len,
                                        wasmer_value_t *results,
                                        unsigned int results_len);

/// Sets the params buffer to the parameter types of the given wasmer_export_func_t
///
/// Returns `wasmer_result_t::WASMER_OK` upon success.
///
/// Returns `wasmer_result_t::WASMER_ERROR` upon failure. Use `wasmer_last_error_length`
/// and `wasmer_last_error_message` to get an error message.
wasmer_result_t wasmer_export_func_params(const wasmer_export_func_t *func,
                                          wasmer_value_tag *params,
                                          uint32_t params_len);

/// Sets the result parameter to the arity of the params of the wasmer_export_func_t
///
/// Returns `wasmer_result_t::WASMER_OK` upon success.
///
/// Returns `wasmer_result_t::WASMER_ERROR` upon failure. Use `wasmer_last_error_length`
/// and `wasmer_last_error_message` to get an error message.
wasmer_result_t wasmer_export_func_params_arity(const wasmer_export_func_t *func, uint32_t *result);

/// Sets the returns buffer to the parameter types of the given wasmer_export_func_t
///
/// Returns `wasmer_result_t::WASMER_OK` upon success.
///
/// Returns `wasmer_result_t::WASMER_ERROR` upon failure. Use `wasmer_last_error_length`
/// and `wasmer_last_error_message` to get an error message.
wasmer_result_t wasmer_export_func_returns(const wasmer_export_func_t *func,
                                           wasmer_value_tag *returns,
                                           uint32_t returns_len);

/// Sets the result parameter to the arity of the returns of the wasmer_export_func_t
///
/// Returns `wasmer_result_t::WASMER_OK` upon success.
///
/// Returns `wasmer_result_t::WASMER_ERROR` upon failure. Use `wasmer_last_error_length`
/// and `wasmer_last_error_message` to get an error message.
wasmer_result_t wasmer_export_func_returns_arity(const wasmer_export_func_t *func,
                                                 uint32_t *result);

/// Gets wasmer_export kind
wasmer_import_export_kind wasmer_export_kind(wasmer_export_t *export_);

/// Gets name from wasmer_export
wasmer_byte_array wasmer_export_name(wasmer_export_t *export_);

/// Gets export func from export
const wasmer_export_func_t *wasmer_export_to_func(const wasmer_export_t *export_);

/// Gets a memory pointer from an export pointer.
///
/// Returns `wasmer_result_t::WASMER_OK` upon success.
///
/// Returns `wasmer_result_t::WASMER_ERROR` upon failure. Use `wasmer_last_error_length`
/// and `wasmer_last_error_message` to get an error message.
wasmer_result_t wasmer_export_to_memory(const wasmer_export_t *export_, wasmer_memory_t **memory);

/// Frees the memory for the given exports.
///
/// Check the `wasmer_instance_exports()` function to get a complete
/// example.
///
/// If `exports` is a null pointer, this function does nothing.
///
/// Example:
///
/// ```c
/// // Get some exports.
/// wasmer_exports_t *exports = NULL;
/// wasmer_instance_exports(instance, &exports);
///
/// // Destroy the exports.
/// wasmer_exports_destroy(exports);
/// ```
void wasmer_exports_destroy(wasmer_exports_t *exports);

/// Gets wasmer_export by index
wasmer_export_t *wasmer_exports_get(wasmer_exports_t *exports, int idx);

/// Gets the length of the exports
int wasmer_exports_len(wasmer_exports_t *exports);

/// Frees memory for the given Global
void wasmer_global_destroy(wasmer_global_t *global);

/// Gets the value stored by the given Global
wasmer_value_t wasmer_global_get(wasmer_global_t *global);

/// Returns a descriptor (type, mutability) of the given Global
wasmer_global_descriptor_t wasmer_global_get_descriptor(wasmer_global_t *global);

/// Creates a new Global and returns a pointer to it.
/// The caller owns the object and should call `wasmer_global_destroy` to free it.
wasmer_global_t *wasmer_global_new(wasmer_value_t value, bool mutable_);

/// Sets the value stored by the given Global
void wasmer_global_set(wasmer_global_t *global, wasmer_value_t value);

/// Gets export descriptor kind
wasmer_import_export_kind wasmer_import_descriptor_kind(wasmer_import_descriptor_t *export_);

/// Gets module name for the import descriptor
wasmer_byte_array wasmer_import_descriptor_module_name(wasmer_import_descriptor_t *import_descriptor);

/// Gets name for the import descriptor
wasmer_byte_array wasmer_import_descriptor_name(wasmer_import_descriptor_t *import_descriptor);

/// Gets import descriptors for the given module
///
/// The caller owns the object and should call `wasmer_import_descriptors_destroy` to free it.
void wasmer_import_descriptors(const wasmer_module_t *module,
                               wasmer_import_descriptors_t **import_descriptors);

/// Frees the memory for the given import descriptors
void wasmer_import_descriptors_destroy(wasmer_import_descriptors_t *import_descriptors);

/// Gets import descriptor by index
wasmer_import_descriptor_t *wasmer_import_descriptors_get(wasmer_import_descriptors_t *import_descriptors,
                                                          unsigned int idx);

/// Gets the length of the import descriptors
unsigned int wasmer_import_descriptors_len(wasmer_import_descriptors_t *exports);

/// Frees memory for the given Func
void wasmer_import_func_destroy(wasmer_import_func_t *func);

/// Creates new host function, aka imported function. `func` is a
/// function pointer, where the first argument is the famous `vm::Ctx`
/// (in Rust), or `wasmer_instance_context_t` (in C). All arguments
/// must be typed with compatible WebAssembly native types:
///
/// | WebAssembly type | C/C++ type |
/// | ---------------- | ---------- |
/// | `i32`            | `int32_t`  |
/// | `i64`            | `int64_t`  |
/// | `f32`            | `float`    |
/// | `f64`            | `double`   |
///
/// The function pointer must have a lifetime greater than the
/// WebAssembly instance lifetime.
///
/// The caller owns the object and should call
/// `wasmer_import_func_destroy` to free it.
wasmer_import_func_t *wasmer_import_func_new(void (*func)(void *data),
                                             const wasmer_value_tag *params,
                                             unsigned int params_len,
                                             const wasmer_value_tag *returns,
                                             unsigned int returns_len);

/// Sets the params buffer to the parameter types of the given wasmer_import_func_t
///
/// Returns `wasmer_result_t::WASMER_OK` upon success.
///
/// Returns `wasmer_result_t::WASMER_ERROR` upon failure. Use `wasmer_last_error_length`
/// and `wasmer_last_error_message` to get an error message.
wasmer_result_t wasmer_import_func_params(const wasmer_import_func_t *func,
                                          wasmer_value_tag *params,
                                          unsigned int params_len);

/// Sets the result parameter to the arity of the params of the wasmer_import_func_t
///
/// Returns `wasmer_result_t::WASMER_OK` upon success.
///
/// Returns `wasmer_result_t::WASMER_ERROR` upon failure. Use `wasmer_last_error_length`
/// and `wasmer_last_error_message` to get an error message.
wasmer_result_t wasmer_import_func_params_arity(const wasmer_import_func_t *func, uint32_t *result);

/// Sets the returns buffer to the parameter types of the given wasmer_import_func_t
///
/// Returns `wasmer_result_t::WASMER_OK` upon success.
///
/// Returns `wasmer_result_t::WASMER_ERROR` upon failure. Use `wasmer_last_error_length`
/// and `wasmer_last_error_message` to get an error message.
wasmer_result_t wasmer_import_func_returns(const wasmer_import_func_t *func,
                                           wasmer_value_tag *returns,
                                           unsigned int returns_len);

/// Sets the result parameter to the arity of the returns of the wasmer_import_func_t
///
/// Returns `wasmer_result_t::WASMER_OK` upon success.
///
/// Returns `wasmer_result_t::WASMER_ERROR` upon failure. Use `wasmer_last_error_length`
/// and `wasmer_last_error_message` to get an error message.
wasmer_result_t wasmer_import_func_returns_arity(const wasmer_import_func_t *func,
                                                 uint32_t *result);

/// Frees memory of the given ImportObject
void wasmer_import_object_destroy(wasmer_import_object_t *import_object);

/// Extends an existing import object with new imports
wasmer_result_t wasmer_import_object_extend(wasmer_import_object_t *import_object,
                                            const wasmer_import_t *imports,
                                            unsigned int imports_len);

/// Gets an entry from an ImportObject at the name and namespace.
/// Stores `name`, `namespace`, and `import_export_value` in `import`.
/// Thus these must remain valid for the lifetime of `import`.
///
/// The caller owns all data involved.
/// `import_export_value` will be written to based on `tag`.
wasmer_result_t wasmer_import_object_get_import(const wasmer_import_object_t *import_object,
                                                wasmer_byte_array namespace_,
                                                wasmer_byte_array name,
                                                wasmer_import_t *import,
                                                wasmer_import_export_value *import_export_value,
                                                uint32_t tag);

/// Frees the memory allocated in `wasmer_import_object_iter_next`
///
/// This function does not free the memory in `wasmer_import_object_t`;
/// it only frees memory allocated while querying a `wasmer_import_object_t`.
void wasmer_import_object_imports_destroy(wasmer_import_t *imports, uint32_t imports_len);

/// Returns true if further calls to `wasmer_import_object_iter_next` will
/// not return any new data
bool wasmer_import_object_iter_at_end(wasmer_import_object_iter_t *import_object_iter);

/// Frees the memory allocated by `wasmer_import_object_iterate_functions`
void wasmer_import_object_iter_destroy(wasmer_import_object_iter_t *import_object_iter);

/// Writes the next value to `import`.  `WASMER_ERROR` is returned if there
/// was an error or there's nothing left to return.
///
/// To free the memory allocated here, pass the import to `wasmer_import_object_imports_destroy`.
/// To check if the iterator is done, use `wasmer_import_object_iter_at_end`.
wasmer_result_t wasmer_import_object_iter_next(wasmer_import_object_iter_t *import_object_iter,
                                               wasmer_import_t *import);

/// Create an iterator over the functions in the import object.
/// Get the next import with `wasmer_import_object_iter_next`
/// Free the iterator with `wasmer_import_object_iter_destroy`
wasmer_import_object_iter_t *wasmer_import_object_iterate_functions(const wasmer_import_object_t *import_object);

/// Creates a new empty import object.
/// See also `wasmer_import_object_append`
wasmer_import_object_t *wasmer_import_object_new();

/// Calls an exported function of a WebAssembly instance by `name`
/// with the provided parameters. The exported function results are
/// stored on the provided `results` pointer.
///
/// This function returns `wasmer_result_t::WASMER_OK` upon success,
/// `wasmer_result_t::WASMER_ERROR` otherwise. You can use
/// `wasmer_last_error_message()` to get the generated error message.
///
/// Potential errors are the following:
///
///   * `instance` is a null pointer,
///   * `name` is a null pointer,
///   * `params` is a null pointer.
///
/// Example of calling an exported function that needs two parameters, and returns one value:
///
/// ```c
/// // First argument.
/// wasmer_value_t argument_one = {
///     .tag = WASM_I32,
///     .value.I32 = 3,
/// };
///
/// // Second argument.
/// wasmer_value_t argument_two = {
///     .tag = WASM_I32,
///     .value.I32 = 4,
/// };
///
/// // First result.
/// wasmer_value_t result_one;
///
/// // All arguments and results.
/// wasmer_value_t arguments[] = {argument_one, argument_two};
/// wasmer_value_t results[]   = {result_one};
///
/// wasmer_result_t call_result = wasmer_instance_call(
///     instance,  // instance pointer
///     "sum",     // the exported function name
///     arguments, // the arguments
///     2,         // the number of arguments
///     results,   // the results
///     1          // the number of results
/// );
///
/// if (call_result == WASMER_OK) {
///     printf("Result is: %d\n", results[0].value.I32);
/// }
/// ```
wasmer_result_t wasmer_instance_call(wasmer_instance_t *instance,
                                     const char *name,
                                     const wasmer_value_t *params,
                                     uint32_t params_len,
                                     wasmer_value_t *results,
                                     uint32_t results_len);

/// Gets the data that can be hold by an instance.
///
/// This function is complementary of
/// `wasmer_instance_context_data_set()`. Please read its
/// documentation. You can also read the documentation of
/// `wasmer_instance_context_t` to get other examples.
///
/// This function returns nothing if `ctx` is a null pointer.
void *wasmer_instance_context_data_get(const wasmer_instance_context_t *ctx);

/// Sets the data that can be hold by an instance context.
///
/// An instance context (represented by the opaque
/// `wasmer_instance_context_t` structure) can hold user-defined
/// data. This function sets the data. This function is complementary
/// of `wasmer_instance_context_data_get()`.
///
/// This function does nothing if `instance` is a null pointer.
///
/// Example:
///
/// ```c
/// // Define your own data.
/// typedef struct {
///     // …
/// } my_data;
///
/// // Allocate them and set them on the given instance.
/// my_data *data = malloc(sizeof(my_data));
/// data->… = …;
/// wasmer_instance_context_data_set(instance, (void*) data);
///
/// // You can read your data.
/// {
///     my_data *data = (my_data*) wasmer_instance_context_data_get(wasmer_instance_context_get(instance));
///     // …
/// }
/// ```
void wasmer_instance_context_data_set(wasmer_instance_t *instance,
                                      void *data_ptr);

/// Returns the instance context. Learn more by looking at the
/// `wasmer_instance_context_t` struct.
///
/// This function returns `null` if `instance` is a null pointer.
///
/// Example:
///
/// ```c
/// const wasmer_instance_context_get *context = wasmer_instance_context_get(instance);
/// my_data *data = (my_data *) wasmer_instance_context_data_get(context);
/// // Do something with `my_data`.
/// ```
///
/// It is often useful with `wasmer_instance_context_data_set()`.
const wasmer_instance_context_t *wasmer_instance_context_get(wasmer_instance_t *instance);

/// Gets the `memory_idx`th memory of the instance.
///
/// Note that the index is always `0` until multiple memories are supported.
///
/// This function is mostly used inside host functions (aka imported
/// functions) to read the instance memory.
///
/// Example of a _host function_ that reads and prints a string based on a pointer and a length:
///
/// ```c
/// void print_string(const wasmer_instance_context_t *context, int32_t pointer, int32_t length) {
///     // Get the 0th memory.
///     const wasmer_memory_t *memory = wasmer_instance_context_memory(context, 0);
///
///     // Get the memory data as a pointer.
///     uint8_t *memory_bytes = wasmer_memory_data(memory);
///
///     // Print what we assumed to be a string!
///     printf("%.*s", length, memory_bytes + pointer);
/// }
/// ```
const wasmer_memory_t *wasmer_instance_context_memory(const wasmer_instance_context_t *ctx,
                                                      uint32_t _memory_idx);

/// Frees memory for the given `wasmer_instance_t`.
///
/// Check the `wasmer_instantiate()` function to get a complete
/// example.
///
/// If `instance` is a null pointer, this function does nothing.
///
/// Example:
///
/// ```c
/// // Get an instance.
/// wasmer_instance_t *instance = NULL;
/// wasmer_instantiate(&instance, bytes, bytes_length, imports, 0);
///
/// // Destroy the instance.
/// wasmer_instance_destroy(instance);
/// ```
void wasmer_instance_destroy(wasmer_instance_t *instance);

/// Gets all the exports of the given WebAssembly instance.
///
/// This function stores a Rust vector of exports into `exports` as an
/// opaque pointer of kind `wasmer_exports_t`.
///
/// As is, you can do anything with `exports` except using the
/// companion functions, like `wasmer_exports_len()`,
/// `wasmer_exports_get()` or `wasmer_export_kind()`. See the example below.
///
/// **Warning**: The caller owns the object and should call
/// `wasmer_exports_destroy()` to free it.
///
/// Example:
///
/// ```c
/// // Get the exports.
/// wasmer_exports_t *exports = NULL;
/// wasmer_instance_exports(instance, &exports);
///
/// // Get the number of exports.
/// int exports_length = wasmer_exports_len(exports);
/// printf("Number of exports: %d\n", exports_length);
///
/// // Read the first export.
/// wasmer_export_t *export = wasmer_exports_get(exports, 0);
///
/// // Get the kind of the export.
/// wasmer_import_export_kind export_kind = wasmer_export_kind(export);
///
/// // Assert it is a function (why not).
/// assert(export_kind == WASM_FUNCTION);
///
/// // Read the export name.
/// wasmer_byte_array name_bytes = wasmer_export_name(export);
///
/// assert(name_bytes.bytes_len == sizeof("sum") - 1);
/// assert(memcmp(name_bytes.bytes, "sum", sizeof("sum") - 1) == 0);
///
/// // Destroy the exports.
/// wasmer_exports_destroy(exports);
/// ```
void wasmer_instance_exports(wasmer_instance_t *instance, wasmer_exports_t **exports);

/// Creates a new WebAssembly instance from the given bytes and imports.
///
/// The result is stored in the first argument `instance` if
/// successful, i.e. when the function returns
/// `wasmer_result_t::WASMER_OK`. Otherwise
/// `wasmer_result_t::WASMER_ERROR` is returned, and
/// `wasmer_last_error_length()` with `wasmer_last_error_message()` must
/// be used to read the error message.
///
/// The caller is responsible to free the instance with
/// `wasmer_instance_destroy()`.
///
/// Example:
///
/// ```c
/// // 1. Read a WebAssembly module from a file.
/// FILE *file = fopen("sum.wasm", "r");
/// fseek(file, 0, SEEK_END);
/// long bytes_length = ftell(file);
/// uint8_t *bytes = malloc(bytes_length);
/// fseek(file, 0, SEEK_SET);
/// fread(bytes, 1, bytes_length, file);
/// fclose(file);
///
/// // 2. Declare the imports (here, none).
/// wasmer_import_t imports[] = {};
///
/// // 3. Instantiate the WebAssembly module.
/// wasmer_instance_t *instance = NULL;
/// wasmer_result_t result = wasmer_instantiate(&instance, bytes, bytes_length, imports, 0);
///
/// // 4. Check for errors.
/// if (result != WASMER_OK) {
///     int error_length = wasmer_last_error_length();
///     char *error = malloc(error_length);
///     wasmer_last_error_message(error, error_length);
///     // Do something with `error`…
/// }
///
/// // 5. Free the memory!
/// wasmer_instance_destroy(instance);
/// ```
wasmer_result_t wasmer_instantiate(wasmer_instance_t **instance,
                                   uint8_t *wasm_bytes,
                                   uint32_t wasm_bytes_len,
                                   wasmer_import_t *imports,
                                   int imports_len);

/// Gets the length in bytes of the last error if any.
///
/// This can be used to dynamically allocate a buffer with the correct number of
/// bytes needed to store a message.
///
/// See `wasmer_last_error_message()` to get a full example.
int wasmer_last_error_length();

/// Gets the last error message if any into the provided buffer
/// `buffer` up to the given `length`.
///
/// The `length` parameter must be large enough to store the last
/// error message. Ideally, the value should come from
/// `wasmer_last_error_length()`.
///
/// The function returns the length of the string in bytes, `-1` if an
/// error occurs. Potential errors are:
///
///  * The buffer is a null pointer,
///  * The buffer is too small to hold the error message.
///
/// Note: The error message always has a trailing NUL character.
///
/// Example:
///
/// ```c
/// int error_length = wasmer_last_error_length();
///
/// if (error_length > 0) {
///     char *error_message = malloc(error_length);
///     wasmer_last_error_message(error_message, error_length);
///     printf("Error message: `%s`\n", error_message);
/// } else {
///     printf("No error message\n");
/// }
/// ```
int wasmer_last_error_message(char *buffer, int length);

/// Gets a pointer to the beginning of the contiguous memory data
/// bytes.
///
/// The function returns `NULL` if `memory` is a null pointer.
///
/// Note that when the memory grows, it can be reallocated, and thus
/// the returned pointer can be invalidated.
///
/// Example:
///
/// ```c
/// uint8_t *memory_data = wasmer_memory_data(memory);
/// char *str = (char*) malloc(sizeof(char) * 7);
///
/// for (uint32_t nth = 0; nth < 7; ++nth) {
///     str[nth] = (char) memory_data[nth];
/// }
/// ```
uint8_t *wasmer_memory_data(const wasmer_memory_t *memory);

/// Gets the size in bytes of the memory data.
///
/// This function returns 0 if `memory` is a null pointer.
///
/// Example:
///
/// ```c
/// uint32_t memory_data_length = wasmer_memory_data_length(memory);
/// ```
uint32_t wasmer_memory_data_length(const wasmer_memory_t *memory);

/// Frees memory for the given `wasmer_memory_t`.
///
/// Check the `wasmer_memory_new()` function to get a complete
/// example.
///
/// If `memory` is a null pointer, this function does nothing.
///
/// Example:
///
/// ```c
/// // Get a memory.
/// wasmer_memory_t *memory = NULL;
/// wasmer_result_t result = wasmer_memory_new(&memory, memory_descriptor);
///
/// // Destroy the memory.
/// wasmer_memory_destroy(memory);
/// ```
void wasmer_memory_destroy(wasmer_memory_t *memory);

/// Grows a memory by the given number of pages (of 65Kb each).
///
/// The functions return `wasmer_result_t::WASMER_OK` upon success,
/// `wasmer_result_t::WASMER_ERROR` otherwise. Use
/// `wasmer_last_error_length()` with `wasmer_last_error_message()` to
/// read the error message.
///
/// Example:
///
/// ```c
/// wasmer_result_t result = wasmer_memory_grow(memory, 10);
///
/// if (result != WASMER_OK) {
///     // …
/// }
/// ```
wasmer_result_t wasmer_memory_grow(wasmer_memory_t *memory, uint32_t delta);

/// Reads the current length (in pages) of the given memory.
///
/// The function returns zero if `memory` is a null pointer.
///
/// Example:
///
/// ```c
/// uint32_t memory_length = wasmer_memory_length(memory);
///
/// printf("Memory pages length: %d\n", memory_length);
/// ```
uint32_t wasmer_memory_length(const wasmer_memory_t *memory);

/// Creates a new empty WebAssembly memory for the given descriptor.
///
/// The result is stored in the first argument `memory` if successful,
/// i.e. when the function returns
/// `wasmer_result_t::WASMER_OK`. Otherwise,
/// `wasmer_result_t::WASMER_ERROR` is returned, and
/// `wasmer_last_error_length()` with `wasmer_last_error_message()`
/// must be used to read the error message.
///
/// The caller owns the memory and is responsible to free it with
/// `wasmer_memory_destroy()`.
///
/// Example:
///
/// ```c
/// // 1. The memory object.
/// wasmer_memory_t *memory = NULL;
///
/// // 2. The memory descriptor.
/// wasmer_limits_t memory_descriptor = {
///     .min = 10,
///     .max = {
///         .has_some = true,
///         .some = 15,
///     },
/// };
///
/// // 3. Initialize the memory.
/// wasmer_result_t result = wasmer_memory_new(&memory, memory_descriptor);
///
/// if (result != WASMER_OK) {
///     int error_length = wasmer_last_error_length();
///     char *error = malloc(error_length);
///     wasmer_last_error_message(error, error_length);
///     // Do something with `error`…
/// }
///
/// // 4. Free the memory!
/// wasmer_memory_destroy(memory);
/// ```
wasmer_result_t wasmer_memory_new(wasmer_memory_t **memory, wasmer_limits_t limits);

/// Deserialize the given serialized module.
///
/// Returns `wasmer_result_t::WASMER_OK` upon success.
///
/// Returns `wasmer_result_t::WASMER_ERROR` upon failure. Use `wasmer_last_error_length`
/// and `wasmer_last_error_message` to get an error message.
wasmer_result_t wasmer_module_deserialize(wasmer_module_t **module,
                                          const wasmer_serialized_module_t *serialized_module);

/// Frees memory for the given Module
void wasmer_module_destroy(wasmer_module_t *module);

/// Given:
/// * A prepared `wasmer` import-object
/// * A compiled wasmer module
///
/// Instantiates a wasmer instance
wasmer_result_t wasmer_module_import_instantiate(wasmer_instance_t **instance,
                                                 const wasmer_module_t *module,
                                                 const wasmer_import_object_t *import_object);

/// Creates a new Instance from the given module and imports.
///
/// Returns `wasmer_result_t::WASMER_OK` upon success.
///
/// Returns `wasmer_result_t::WASMER_ERROR` upon failure. Use `wasmer_last_error_length`
/// and `wasmer_last_error_message` to get an error message.
wasmer_result_t wasmer_module_instantiate(const wasmer_module_t *module,
                                          wasmer_instance_t **instance,
                                          wasmer_import_t *imports,
                                          int imports_len);

/// Serialize the given Module.
///
/// The caller owns the object and should call `wasmer_serialized_module_destroy` to free it.
///
/// Returns `wasmer_result_t::WASMER_OK` upon success.
///
/// Returns `wasmer_result_t::WASMER_ERROR` upon failure. Use `wasmer_last_error_length`
/// and `wasmer_last_error_message` to get an error message.
wasmer_result_t wasmer_module_serialize(wasmer_serialized_module_t **serialized_module_out,
                                        const wasmer_module_t *module);

/// Get bytes of the serialized module.
wasmer_byte_array wasmer_serialized_module_bytes(const wasmer_serialized_module_t *serialized_module);

/// Frees memory for the given serialized Module.
void wasmer_serialized_module_destroy(wasmer_serialized_module_t *serialized_module);

/// Transform a sequence of bytes into a serialized module.
///
/// The caller owns the object and should call `wasmer_serialized_module_destroy` to free it.
///
/// Returns `wasmer_result_t::WASMER_OK` upon success.
///
/// Returns `wasmer_result_t::WASMER_ERROR` upon failure. Use `wasmer_last_error_length`
/// and `wasmer_last_error_message` to get an error message.
wasmer_result_t wasmer_serialized_module_from_bytes(wasmer_serialized_module_t **serialized_module,
                                                    const uint8_t *serialized_module_bytes,
                                                    uint32_t serialized_module_bytes_length);

/// Frees memory for the given Table
void wasmer_table_destroy(wasmer_table_t *table);

/// Grows a Table by the given number of elements.
///
/// Returns `wasmer_result_t::WASMER_OK` upon success.
///
/// Returns `wasmer_result_t::WASMER_ERROR` upon failure. Use `wasmer_last_error_length`
/// and `wasmer_last_error_message` to get an error message.
wasmer_result_t wasmer_table_grow(wasmer_table_t *table, uint32_t delta);

/// Returns the current length of the given Table
uint32_t wasmer_table_length(wasmer_table_t *table);

/// Creates a new Table for the given descriptor and initializes the given
/// pointer to pointer to a pointer to the new Table.
///
/// The caller owns the object and should call `wasmer_table_destroy` to free it.
///
/// Returns `wasmer_result_t::WASMER_OK` upon success.
///
/// Returns `wasmer_result_t::WASMER_ERROR` upon failure. Use `wasmer_last_error_length`
/// and `wasmer_last_error_message` to get an error message.
wasmer_result_t wasmer_table_new(wasmer_table_t **table, wasmer_limits_t limits);

/// Stop the execution of a host function, aka imported function. The
/// function must be used _only_ inside a host function.
///
/// The pointer to `wasmer_instance_context_t` is received by the host
/// function as its first argument. Just passing it to `ctx` is fine.
///
/// The error message must have a greater lifetime than the host
/// function itself since the error is read outside the host function
/// with `wasmer_last_error_message`.
///
/// This function returns `wasmer_result_t::WASMER_ERROR` if `ctx` or
/// `error_message` are null.
///
/// This function never returns otherwise.
wasmer_result_t wasmer_trap(const wasmer_instance_context_t *_ctx, const char *error_message);

/// Validates a sequence of bytes hoping it represents a valid WebAssembly module.
///
/// The function returns true if the bytes are valid, false otherwise.
///
/// Example:
///
/// ```c
/// bool result = wasmer_validate(bytes, bytes_length);
///
/// if (false == result) {
///     // Do something…
/// }
/// ```
bool wasmer_validate(const uint8_t *wasm_bytes, uint32_t wasm_bytes_len);

#if defined(WASMER_WASI_ENABLED)
/// Convenience function that creates a WASI import object with no arguments,
/// environment variables, preopened files, or mapped directories.
///
/// This function is the same as calling [`wasmer_wasi_generate_import_object`] with all
/// empty values.
wasmer_import_object_t *wasmer_wasi_generate_default_import_object();
#endif

#if defined(WASMER_WASI_ENABLED)
/// Creates a WASI import object.
///
/// This function treats null pointers as empty collections.
/// For example, passing null for a string in `args`, will lead to a zero
/// length argument in that position.
wasmer_import_object_t *wasmer_wasi_generate_import_object(const wasmer_byte_array *args,
                                                           unsigned int args_len,
                                                           const wasmer_byte_array *envs,
                                                           unsigned int envs_len,
                                                           const wasmer_byte_array *preopened_files,
                                                           unsigned int preopened_files_len,
                                                           const wasmer_wasi_map_dir_entry_t *mapped_dirs,
                                                           unsigned int mapped_dirs_len);
#endif

#if defined(WASMER_WASI_ENABLED)
/// Creates a WASI import object for a specific version.
///
/// This function is similar to `wasmer_wasi_generate_import_object`
/// except that the first argument describes the WASI version.
///
/// The version is expected to be of kind `Version`.
wasmer_import_object_t *wasmer_wasi_generate_import_object_for_version(unsigned char version,
                                                                       const wasmer_byte_array *args,
                                                                       unsigned int args_len,
                                                                       const wasmer_byte_array *envs,
                                                                       unsigned int envs_len,
                                                                       const wasmer_byte_array *preopened_files,
                                                                       unsigned int preopened_files_len,
                                                                       const wasmer_wasi_map_dir_entry_t *mapped_dirs,
                                                                       unsigned int mapped_dirs_len);
#endif

#if defined(WASMER_WASI_ENABLED)
/// Find the version of WASI used by the module.
///
/// In case of error, the returned version is `Version::Unknown`.
Version wasmer_wasi_get_version(const wasmer_module_t *module);
#endif

} // extern "C"

#endif // WASMER_H
