//----------------------------------------------------------------------------------------------------------------------
// SaneCppReflection.h - Sane C++ Reflection Library (single file build)
//----------------------------------------------------------------------------------------------------------------------
// Dependencies:       None
// Version:            release/2026/08 (61a46a65f)
// LOC header:         930 (code) + 483 (comments)
// LOC implementation: 0 (code) + 0 (comments)
// Documentation:      https://pagghiu.github.io/SaneCppLibraries
// Source Code:        https://github.com/pagghiu/SaneCppLibraries
//----------------------------------------------------------------------------------------------------------------------
// All copyrights and SPDX information for this library (each amalgamated section has its own copyright attributions):
// Copyright (c) Stefano Cristiano
// SPDX-License-Identifier: MIT
//----------------------------------------------------------------------------------------------------------------------
#if defined(SANE_CPP_REFLECTION_INCLUDED) && !defined(SANE_CPP_REFLECTION_CONTENT_CBE574D32C541AD2)
#error "SaneCppReflection was already included from a different single-file version"
#endif

#if (defined(SANE_CPP_REFLECTION_HEADER) || defined(SANE_CPP_REFLECTION_IMPLEMENTATION)) && !defined(SANE_CPP_REFLECTION_INCLUDED)
#error "SaneCppReflection was already included without single-file content markers"
#endif

#if !defined(SANE_CPP_REFLECTION_INCLUDED)
#define SANE_CPP_REFLECTION_INCLUDED 1
#define SANE_CPP_REFLECTION_CONTENT_CBE574D32C541AD2 1
#endif

#if !defined(SANE_CPP_REFLECTION_HEADER)
#define SANE_CPP_REFLECTION_HEADER 1
//----------------------------------------------------------------------------------------------------------------------
// Reflection/Reflection.h
//----------------------------------------------------------------------------------------------------------------------
// Copyright (c) Stefano Cristiano
// SPDX-License-Identifier: MIT

//----------------------------------------------------------------------------------------------------------------------
// Common/CompilerMacrosExport.h
//----------------------------------------------------------------------------------------------------------------------
// Copyright (c) Stefano Cristiano
// SPDX-License-Identifier: MIT
#if defined(SC_FOUNDATION_COMPILER_MACROS_EXPORT_DEFINITION_H)
#if SC_FOUNDATION_COMPILER_MACROS_EXPORT_DEFINITION_H != 1
#error "CompilerMacrosExport.h has been included multiple times in different versions."
#endif
#else
#define SC_FOUNDATION_COMPILER_MACROS_EXPORT_DEFINITION_H 1 // Increment to indicate a new version of the file

/// Define compiler-specific visibility macros.
#if defined(_MSC_VER)
#define SC_COMPILER_SYMBOL_IMPORT __declspec(dllimport)
#define SC_COMPILER_SYMBOL_EXPORT __declspec(dllexport)
#define SC_COMPILER_SYMBOL_HIDDEN
#else
#define SC_COMPILER_SYMBOL_IMPORT
#define SC_COMPILER_SYMBOL_EXPORT __attribute__((visibility("default")))
#define SC_COMPILER_SYMBOL_HIDDEN __attribute__((visibility("hidden")))
#endif

#if defined(SC_PLUGIN_LIBRARY)
#define SC_COMPILER_EXTERN extern
#else
#define SC_COMPILER_EXTERN
#endif

#if defined(SC_PLUGIN_LIBRARY)
#define SC_COMPILER_LIBRARY_EXPORT_0 SC_COMPILER_SYMBOL_IMPORT
#define SC_COMPILER_LIBRARY_EXPORT_1 SC_COMPILER_SYMBOL_IMPORT
#else
#define SC_COMPILER_LIBRARY_EXPORT_0 SC_COMPILER_SYMBOL_HIDDEN
#define SC_COMPILER_LIBRARY_EXPORT_1 SC_COMPILER_SYMBOL_EXPORT
#endif
#define SC_COMPILER_LIBRARY_EXPORT(value)      SC_COMPILER_LIBRARY_EXPORT_IMPL(value)
#define SC_COMPILER_LIBRARY_EXPORT_IMPL(value) SC_COMPILER_LIBRARY_EXPORT_##value

#ifndef SC_EXPORT_LIBRARY_FOUNDATION
#define SC_EXPORT_LIBRARY_FOUNDATION 0
#endif
#define SC_FOUNDATION_EXPORT SC_COMPILER_LIBRARY_EXPORT(SC_EXPORT_LIBRARY_FOUNDATION)

#endif // SC_FOUNDATION_COMPILER_MACROS_EXPORT_DEFINITION_H

//----------------------------------------------------------------------------------------------------------------------
// Common/CompilerOffsetOf.h
//----------------------------------------------------------------------------------------------------------------------
// Copyright (c) Stefano Cristiano
// SPDX-License-Identifier: MIT
#ifdef SC_FOUNDATION_COMPILER_OFFSET_OF_DEFINITION_H
#if SC_FOUNDATION_COMPILER_OFFSET_OF_DEFINITION_H != 1
#error "CompilerOffsetOf.h has been included multiple times in different versions."
#endif
#else
#define SC_FOUNDATION_COMPILER_OFFSET_OF_DEFINITION_H 1 // Increment to indicate a new version of the file

/// Returns offset of Class::Field in bytes
#define SC_COMPILER_OFFSETOF(Class, Field)            __builtin_offsetof(Class, Field)

namespace SC
{
// clang-format off
template <int offset, typename T, typename R> T& fieldOffset(R& object) { return *reinterpret_cast<T*>(reinterpret_cast<char*>(&object) - offset); }
#define SC_COMPILER_FIELD_OFFSET(Class, Field, Value)  SC::fieldOffset<SC_COMPILER_OFFSETOF(Class, Field), Class, decltype(Class::Field)>(Value);
// clang-format on
} // namespace SC

/// Disables invalid-offsetof gcc and clang warning
#if defined(__clang__)
#define SC_COMPILER_WARNING_PUSH_OFFSETOF                                                                              \
    _Pragma("clang diagnostic push");                                                                                  \
    _Pragma("clang diagnostic ignored \"-Winvalid-offsetof\"");
#define SC_COMPILER_WARNING_POP_OFFSETOF _Pragma("clang diagnostic pop")
#elif defined(__GNUC__)
#define SC_COMPILER_WARNING_PUSH_OFFSETOF                                                                              \
    _Pragma("GCC diagnostic push");                                                                                    \
    _Pragma("GCC diagnostic ignored \"-Winvalid-offsetof\"");
#define SC_COMPILER_WARNING_POP_OFFSETOF _Pragma("GCC diagnostic pop")
#else
#define SC_COMPILER_WARNING_PUSH_OFFSETOF _Pragma("warning(push)")
#define SC_COMPILER_WARNING_POP_OFFSETOF  _Pragma("warning(pop)")
#endif
#endif // SC_FOUNDATION_COMPILER_OFFSET_OF_DEFINITION_H

//----------------------------------------------------------------------------------------------------------------------
// Common/PrimitiveDefinitions.h
//----------------------------------------------------------------------------------------------------------------------
// Copyright (c) Stefano Cristiano
// SPDX-License-Identifier: MIT
#ifdef SC_FOUNDATION_PRIMITIVE_TYPES_DEFINITION_H
#if SC_FOUNDATION_PRIMITIVE_TYPES_DEFINITION_H != 1
#error "PrimitiveTypesDefinition.h has been included multiple times in different versions."
#endif
#else
#define SC_FOUNDATION_PRIMITIVE_TYPES_DEFINITION_H 1 // Increment to indicate a new version of the file

namespace SC
{
// clang-format off
//! @addtogroup group_foundation_utility
//! @{
#if defined(_WIN32) || defined(_WIN64)
using native_char_t = wchar_t;

using uint8_t  = unsigned char;
using uint16_t = unsigned short;
using uint32_t = unsigned int;
using uint64_t = unsigned long long;

using int8_t  = signed char;
using int16_t = short;
using int32_t = int;
using int64_t = long long;
#else
using native_char_t = char; ///< The native char for the platform (wchar_t (4 bytes) on Windows, char (1 byte) everywhere else )

using uint8_t  = unsigned char;     ///< Platform independent (1) byte unsigned int
using uint16_t = unsigned short;    ///< Platform independent (2) bytes unsigned int
using uint32_t = unsigned int;      ///< Platform independent (4) bytes unsigned int
#if defined(__linux__)
using uint64_t = unsigned long int; ///< Platform independent (8) bytes unsigned int
#else
using uint64_t = unsigned long long;///< Platform independent (8) bytes unsigned int
#endif
using int8_t  = signed char;        ///< Platform independent (1) byte signed int
using int16_t = short;              ///< Platform independent (2) bytes signed int
using int32_t = int;                ///< Platform independent (4) bytes signed int
#if defined(__linux__)
using int64_t = signed long int;    ///< Platform independent (8) bytes signed int
#else
using int64_t = long long;          ///< Platform independent (8) bytes signed int
#endif
#endif

using size_t  = decltype(sizeof(0));                                                 ///< Platform independent unsigned size type
using ssize_t = decltype(static_cast<char*>(nullptr) - static_cast<char*>(nullptr)); ///< Platform independent signed size type

/// @brief A vocabulary type representing a time interval in milliseconds since epoch
struct TimeMs
{
    int64_t milliseconds = 0;
};
//! @}
} // namespace SC

#endif // SC_FOUNDATION_PRIMITIVE_TYPES_DEFINITION_H

#ifndef SC_EXPORT_LIBRARY_REFLECTION
#define SC_EXPORT_LIBRARY_REFLECTION 0
#endif
#define SC_REFLECTION_EXPORT SC_COMPILER_LIBRARY_EXPORT(SC_EXPORT_LIBRARY_REFLECTION)

//----------------------------------------------------------------------------------------------------------------------
// Common/TypeTraits.h
//----------------------------------------------------------------------------------------------------------------------
// Copyright (c) Stefano Cristiano
// SPDX-License-Identifier: MIT
#ifdef SC_FOUNDATION_TYPE_TRAITS_DEFINITION_H
#if SC_FOUNDATION_TYPE_TRAITS_DEFINITION_H != 1
#error "TypeTraits.h has been included multiple times in different versions."
#endif
#else
#define SC_FOUNDATION_TYPE_TRAITS_DEFINITION_H 1 // Increment to indicate a new version of the file

//----------------------------------------------------------------------------------------------------------------------
// Common/CompilerMove.h
//----------------------------------------------------------------------------------------------------------------------
// Copyright (c) Stefano Cristiano
// SPDX-License-Identifier: MIT
#ifdef SC_FOUNDATION_COMPILER_MOVE_DEFINITION_H
#if SC_FOUNDATION_COMPILER_MOVE_DEFINITION_H != 1
#error "CompilerMove.h has been included multiple times in different versions."
#endif
#else
#define SC_FOUNDATION_COMPILER_MOVE_DEFINITION_H 1 // Increment to indicate a new version of the file
namespace SC
{

// clang-format off
namespace TypeTraits
{
//! @addtogroup group_foundation_type_traits
//! @{
template <class T> struct RemoveReference       { using type = T; };
template <class T> struct RemoveReference<T&>   { using type = T; };
template <class T> struct RemoveReference<T&&>  { using type = T; };
/// Determines if a type is an lvalue reference.
template <class T> struct IsLValueReference     { static constexpr bool value = false; };
template <class T> struct IsLValueReference<T&> { static constexpr bool value = true; };
/// Determines if a type is an rvalue reference.
template <class T> struct IsRValueReference     { static constexpr bool value = false; };
template <class T> struct IsRValueReference<T&&>{ static constexpr bool value = true; };
//! @}

} // namespace TypeTraits
//! @defgroup group_foundation_utility Language Utilities
//! @ingroup group_foundation
//! Language Utilities
/// Utility classes allowing common C++ constructs.

//! @addtogroup group_foundation_utility
//! @{

/// Converts an lvalue to an rvalue reference.
template <typename T> constexpr T&& move(T& value) { return static_cast<T&&>(value); }

/// Forwards an lvalue or an rvalue as an rvalue reference.
template <typename T> constexpr T&& forward(typename TypeTraits::RemoveReference<T>::type& value) { return static_cast<T&&>(value); }

/// Forwards an rvalue as an rvalue reference, with a check that it's not an lvalue reference.
template <typename T> constexpr T&& forward(typename TypeTraits::RemoveReference<T>::type&& value)
{
    static_assert(!TypeTraits::IsLValueReference<T>::value, "Forward an rvalue as an lvalue is not allowed");
    return static_cast<T&&>(value);
}

/// Swaps the values of two objects.
template <typename T> constexpr inline void swap(T& t1, T& t2) { T temp = move(t1); t1 = move(t2); t2 = move(temp); }
//! @}
// clang-format on
} // namespace SC

#endif // SC_FOUNDATION_COMPILER_MOVE_DEFINITION_H

//!
//! @defgroup group_foundation Foundation
//! @copybrief library_foundation
//!
//! See @ref library_foundation library page for more details.<br>

//! @defgroup group_foundation_type_traits Type Traits
//! @ingroup group_foundation
//! Type Traits (EnableIf, AddPointer, etc.)

namespace SC
{
/// @brief Template meta-programming helpers
namespace TypeTraits
{
// clang-format off
//! @addtogroup group_foundation_type_traits
//! @{

/// EnableIf conditionally defines a type if a boolean template parameter is true.
template <bool B, class T = void> struct EnableIf {};
template <class T> struct EnableIf<true, T> { using type = T; };

/// IsSame evaluates to `true` if the provided types `T` and `U` are the same, `false` otherwise.
template <typename T, typename U>   struct IsSame       { static constexpr bool value = false; };
template <typename T>               struct IsSame<T, T> { static constexpr bool value = true;  };

/// AddPointer adds a pointer qualification to a type `T` if it is not already a pointer.
template <class T> struct AddPointer      { using type = typename RemoveReference<T>::type*; };

/// RemoveConst removes the const qualification from a type `T`.
template <class T> struct RemoveConst           { using type = T;};
template <class T> struct RemoveConst<const T>  { using type = T; };

/// IsConst evaluates to `true` if the provided type `T` is `const`, `false` otherwise.
template <typename T> struct IsConst            { using type = T; static constexpr bool value = false; };
template <typename T> struct IsConst<const T>   { using type = T; static constexpr bool value = true;  };

/// IsTriviallyCopyable evaluates to `true` if the type `T` can be trivially copied, `false` otherwise.
template <typename T> struct IsTriviallyCopyable { static constexpr bool value = __is_trivially_copyable(T); };

/// Conditional defines a type to be `T` if a boolean value is `true`, `F` otherwise.
template <bool B, class T, class F> struct Conditional { using type = T; };
template <class T, class F>         struct Conditional<false, T, F> { using type = F; };

/// SameConstnessAs modifies type `T` to have the const-qualification of `U`.
template <typename U, typename T> struct SameConstnessAs { using type = typename Conditional<IsConst<U>::value, const T, T>::type; };

//! @}
// clang-format on
} // namespace TypeTraits
} // namespace SC

#endif

//----------------------------------------------------------------------------------------------------------------------
// Reflection/ReflectionFoundation.h
//----------------------------------------------------------------------------------------------------------------------
// Copyright (c) Stefano Cristiano
// SPDX-License-Identifier: MIT
//----------------------------------------------------------------------------------------------------------------------
// Common/CompilerMacrosType.h
//----------------------------------------------------------------------------------------------------------------------
// Copyright (c) Stefano Cristiano
// SPDX-License-Identifier: MIT
#if defined(SC_FOUNDATION_COMPILER_MACROS_TYPE_DEFINITION_H)
#if SC_FOUNDATION_COMPILER_MACROS_TYPE_DEFINITION_H != 1
#error "CompilerMacrosType.h has been included multiple times in different versions."
#endif
#else
#define SC_FOUNDATION_COMPILER_MACROS_TYPE_DEFINITION_H 1 // Increment to indicate a new version of the file

#if __clang__
#define SC_COMPILER_CLANG 1 ///< Flag indicating Clang compiler.
#define SC_COMPILER_GCC   0 ///< Flag indicating GCC compiler.
#define SC_COMPILER_MSVC  0 ///< Flag indicating MSVC compiler.

#if _MSC_VER
#define SC_COMPILER_CLANG_CL 1 ///< Flag indicating Clang-CL (MSVC) compiler.
#else
#define SC_COMPILER_CLANG_CL 0 ///< Flag indicating non-Clang-CL (MSVC) compiler.
#endif

#elif _MSC_VER
#define SC_COMPILER_CLANG    0 ///< Flag indicating Clang compiler.
#define SC_COMPILER_GCC      0 ///< Flag indicating GCC compiler.
#define SC_COMPILER_MSVC     1 ///< Flag indicating MSVC compiler.
#define SC_COMPILER_CLANG_CL 0 ///< Flag indicating non-Clang-CL (MSVC) compiler.

#else
#define SC_COMPILER_CLANG    0 ///< Flag indicating Clang compiler.
#define SC_COMPILER_GCC      1 ///< Flag indicating GCC compiler.
#define SC_COMPILER_MSVC     0 ///< Flag indicating MSVC compiler.
#define SC_COMPILER_CLANG_CL 0 ///< Flag indicating non-Clang-CL (MSVC) compiler.
#endif

#if defined(__FILC__)
#define SC_COMPILER_FILC 1
#else
#define SC_COMPILER_FILC 0
#endif

#endif // SC_FOUNDATION_COMPILER_MACROS_TYPE_DEFINITION_H

//----------------------------------------------------------------------------------------------------------------------
// Common/Span.h
//----------------------------------------------------------------------------------------------------------------------
// Copyright (c) Stefano Cristiano
// SPDX-License-Identifier: MIT
#ifdef SC_FOUNDATION_SPAN_DEFINITION_H
#if SC_FOUNDATION_SPAN_DEFINITION_H != 1
#error "Span.h has been included multiple times in different versions."
#endif
#else
#define SC_FOUNDATION_SPAN_DEFINITION_H 1 // Increment to indicate a new version of the file

//----------------------------------------------------------------------------------------------------------------------
// Common/InitializerList.h
//----------------------------------------------------------------------------------------------------------------------
// Copyright (c) Stefano Cristiano
// SPDX-License-Identifier: MIT
#ifdef SC_FOUNDATION_INITIALIZER_LIST_DEFINITION_H
#if SC_FOUNDATION_INITIALIZER_LIST_DEFINITION_H != 1
#error "InitializerList.h has been included multiple times in different versions."
#endif
#else
#define SC_FOUNDATION_INITIALIZER_LIST_DEFINITION_H 1 // Increment to indicate a new version of the file

//----------------------------------------------------------------------------------------------------------------------
// Common/CompilerMacrosStdCpp.h
//----------------------------------------------------------------------------------------------------------------------
// Copyright (c) Stefano Cristiano
// SPDX-License-Identifier: MIT
#if defined(SC_FOUNDATION_COMPILER_MACROS_STD_CPP_DEFINITION_H)
#if SC_FOUNDATION_COMPILER_MACROS_STD_CPP_DEFINITION_H != 1
#error "CompilerMacrosStdCpp.h has been included multiple times in different versions."
#endif
#else
#define SC_FOUNDATION_COMPILER_MACROS_STD_CPP_DEFINITION_H 1 // Increment to indicate a new version of the file

#if defined(SC_COMPILER_ENABLE_STD_CPP)
#error                                                                                                                 \
    "SC_COMPILER_ENABLE_STD_CPP has been replaced. Standard C/C++ headers and C++ runtime linkage are enabled by default; define SC_INCLUDE_STD_CPP=0 and SC_PROVIDE_CPP_RUNTIME_SHIMS=1 to request the old no-stdlib mode."
#endif

#if !defined(SC_INCLUDE_STD_CPP)
#define SC_INCLUDE_STD_CPP 1
#elif SC_INCLUDE_STD_CPP != 0 && SC_INCLUDE_STD_CPP != 1
#error "SC_INCLUDE_STD_CPP must be 0 or 1."
#endif

#if !defined(SC_PROVIDE_CPP_RUNTIME_SHIMS)
#define SC_PROVIDE_CPP_RUNTIME_SHIMS 0
#elif SC_PROVIDE_CPP_RUNTIME_SHIMS != 0 && SC_PROVIDE_CPP_RUNTIME_SHIMS != 1
#error "SC_PROVIDE_CPP_RUNTIME_SHIMS must be 0 or 1."
#endif

#endif // SC_FOUNDATION_COMPILER_MACROS_STD_CPP_DEFINITION_H


#if SC_INCLUDE_STD_CPP
#include <initializer_list>
#else
namespace std
{
template <class _Ep>
class initializer_list
{
    using size_t = decltype(sizeof(0));
    const _Ep* __begin_;
    size_t     __size_;

    constexpr initializer_list(const _Ep* __b, size_t __s) noexcept : __begin_(__b), __size_(__s) {}

  public:
    using value_type = _Ep;
    using size_type  = size_t;

    using iterator       = const _Ep*;
    using const_iterator = const _Ep*;

    constexpr initializer_list() noexcept : __begin_(nullptr), __size_(0) {}
    constexpr initializer_list(const _Ep* _First_arg, const _Ep* _Last_arg) noexcept
        : __begin_(_First_arg), __size_(static_cast<size_t>(_Last_arg - _First_arg))
    {}

    constexpr size_t     size() const noexcept { return __size_; }
    constexpr const _Ep* begin() const noexcept { return __begin_; }
    constexpr const _Ep* end() const noexcept { return __begin_ + __size_; }
};

} // namespace std
#endif

#endif


namespace SC
{
template <typename Type>
struct Span;

namespace detail
{
// clang-format off
template<typename U> struct SpanSizeOfType              { static constexpr auto size = sizeof(U);  };
template<>           struct SpanSizeOfType<void>        { static constexpr auto size = 1;          };
template<>           struct SpanSizeOfType<const void>  { static constexpr auto size = 1;          };
// clang-format on
} // namespace detail

//! @addtogroup group_foundation_utility
//! @{

/// @brief View over a contiguous sequence of items (pointer + size in elements).
/// @tparam Type Any type
template <typename Type>
struct SC_FOUNDATION_EXPORT Span
{
  private:
    // clang-format off
    using SizeType = size_t;
    using VoidType = typename TypeTraits::SameConstnessAs<Type, void>::type;
    template <typename U> using TypeIfNotVoid = typename  TypeTraits::EnableIf<not TypeTraits::IsSame<U, VoidType>::value, Type>::type;
    template <typename U> using TypeInitializerList = typename  TypeTraits::EnableIf<TypeTraits::IsConst<U>::value and not TypeTraits::IsSame<U, VoidType>::value, Type>::type;
    template <typename U> using SameType = TypeTraits::IsSame<typename TypeTraits::RemoveConst<U>::type, typename TypeTraits::RemoveConst<Type>::type>;
    template <typename U> using EnableNotVoid = typename TypeTraits::EnableIf<SameType<U>::value and not TypeTraits::IsSame<U, VoidType>::value, bool>::type;
    // clang-format on
    Type*    items;
    SizeType sizeElements;

  public:
    template <size_t N, typename U = Type, EnableNotVoid<U> = true>
    constexpr Span(U (&itemsArray)[N]) : items(itemsArray), sizeElements(N)
    {}

    /// @brief Builds an empty Span
    constexpr Span() : items(nullptr), sizeElements(0) {}

    /// @brief Builds a Span from an array
    /// @param items  pointer to the first member of the array
    /// @param sizeInElements number of elements in in the array
    constexpr Span(Type* items, SizeType sizeInElements) : items(items), sizeElements(sizeInElements) {}

    /// @brief Builds a Span from a single object
    /// @param type A reference to a single object of type Type
    template <typename U = Type>
    constexpr Span(TypeIfNotVoid<U>& type) : items(&type), sizeElements(1)
    {}

    /// @brief Span specialized constructor (mainly used for converting const char* to StringView)
    /// @param list an initializer list of elements
    template <typename U = Type>
    constexpr Span(std::initializer_list<TypeInitializerList<U>> list) : items(nullptr), sizeElements(0)
    {
        // We need this two step initialization to avoid warnings on all compilers
        items        = list.begin();
        sizeElements = list.size();
    }

    // clang-format off
    template <typename U = Type> operator Span<const TypeIfNotVoid<U>>() const { return {items, sizeElements}; }
    template <typename U = Type> operator Span<      TypeIfNotVoid<U>>()       { return {items, sizeElements}; }
    operator Span<const void>() const { return Span<const void>(items, sizeElements * detail::SpanSizeOfType<Type>::size); }
    operator Span<void>() { return Span<void>(items, sizeElements * detail::SpanSizeOfType<Type>::size); }

    /// @brief Constructs a Span reinterpreting memory pointed by object of type `T` as a type `Type`
    template <typename T> [[nodiscard]] static Span<Type> reinterpret_object(T& value) { return {reinterpret_cast<Type*>(&value), sizeof(T) / detail::SpanSizeOfType<Type>::size}; }

    /// @brief Construct a span reinterpreting raw memory (`void*` or `const void*`) to `Type` or `const Type`
    [[nodiscard]] static Span<Type> reinterpret_bytes(VoidType* rawMemory, SizeType sizeInBytes) { return Span(reinterpret_cast<Type*>(rawMemory), sizeInBytes / detail::SpanSizeOfType<Type>::size); }

    /// @brief Reinterprets the current span as an array of the specified type
    template <typename T> [[nodiscard]] Span<const T> reinterpret_as_span_of() const { return Span<const T>(reinterpret_cast<const T*>(items), sizeInBytes() / sizeof(T)); }

    /// @brief Reinterprets the current span as an array of the specified type
    template <typename T> [[nodiscard]] Span<T> reinterpret_as_span_of() { return Span<T>(reinterpret_cast<T*>(items), sizeInBytes() / sizeof(T)); }

    [[nodiscard]] constexpr const Type* begin() const { return items; }
    [[nodiscard]] constexpr const Type* end() const { return items + sizeElements; }
    [[nodiscard]] constexpr const Type* data() const { return items; }
    [[nodiscard]] constexpr Type* begin() { return items; }
    [[nodiscard]] constexpr Type* end() { return items + sizeElements; }
    [[nodiscard]] constexpr Type* data() { return items; }

    [[nodiscard]] constexpr SizeType sizeInElements() const { return sizeElements; }
    [[nodiscard]] constexpr SizeType sizeInBytes() const { return sizeElements * detail::SpanSizeOfType<Type>::size; }

    [[nodiscard]] constexpr bool empty() const { return sizeElements == 0; }

    template <typename U = Type> TypeIfNotVoid<U>& operator[](SizeType idx) {  return items[idx]; }
    template <typename U = Type> const TypeIfNotVoid<U>& operator[](SizeType idx) const { return items[idx]; }
    // clang-format on

    /// @brief Creates another Span, starting at an offset in elements from current Span, until end.
    /// @param offsetInElements Offset in current Span where destination Span will be starting
    /// @param destination Reference to a Span that will hold the resulting sliced span
    /// @return `false` if offsetInElements is bigger to than Span::sizeInElements().
    [[nodiscard]] constexpr bool sliceStart(SizeType offsetInElements, Span& destination) const
    {
        bool valid  = offsetInElements <= sizeElements;
        destination = valid ? Span(items + offsetInElements, sizeElements - offsetInElements) : Span(nullptr, 0);
        return valid;
    }

    /// @brief Creates another Span, starting at an offset in elements from current Span of specified length.
    /// @param offsetInElements Offset in current Span where destination Span will be starting
    /// @param lengthInElements Number of elements to include in destination Span
    /// @param destination Reference to a Span that will hold the resulting sliced span
    /// @return false` if (offsetInElements + lengthInElements) is bigger to than Span::sizeInElements().
    [[nodiscard]] constexpr bool sliceStartLength(SizeType offsetInElements, SizeType lengthInElements,
                                                  Span& destination) const
    {
        bool valid  = offsetInElements + lengthInElements <= sizeElements;
        destination = valid ? Span(items + offsetInElements, lengthInElements) : Span(nullptr, 0);
        return valid;
    }
};
//! @}

} // namespace SC

#endif

namespace SC
{
namespace Reflection
{
//! @addtogroup group_reflection
//! @{

/// @brief A constexpr array
/// @tparam T Item types
/// @tparam N Maximum number of items
template <typename T, uint32_t N>
struct ArrayWithSize
{
    T        values[N] = {};
    uint32_t size      = 0;

    operator Span<const T>() const { return {values, size}; }
    /// @brief Check if array contains given value, and retrieve index where such item exists
    /// @param value Value to look for
    /// @param outIndex if not `nullptr` will receive the index where item has been found
    /// @return `true` if ArrayWithSize contains given value
    [[nodiscard]] constexpr bool contains(T value, uint32_t* outIndex = nullptr) const
    {
        for (uint32_t i = 0; i < size; ++i)
        {
            if (values[i] == value)
            {
                if (outIndex)
                    *outIndex = i;
                return true;
            }
        }
        return false;
    }

    /// @brief Appends another sized array to this one (assuming enough space)
    /// @tparam N2 Maximum size of other array
    /// @param other The array to append to this one
    /// @return `true` if there was enough space to append the other array to this one
    template <uint32_t N2>
    [[nodiscard]] constexpr bool append(const ArrayWithSize<T, N2>& other)
    {
        if (size + other.size >= N)
            return false;
        for (uint32_t i = 0; i < other.size; ++i)
        {
            values[size++] = other.values[i];
        }
        return true;
    }

    /// @brief Append a single item to this array
    /// @param value Value to append
    /// @return `true` if there was enough space for the value
    [[nodiscard]] constexpr bool push_back(const T& value)
    {
        if (size < N)
        {
            values[size++] = value;
            return true;
        }
        return false;
    }

    bool equals(Span<const T> other) const
    {
        return size == other.sizeInElements() and __builtin_memcmp(values, other.data(), other.sizeInBytes()) == 0;
    }
};

/// @brief A writable span of objects
template <typename Type>
struct WritableRange
{
    Type* iterator;
    Type* iteratorEnd;

    constexpr WritableRange(Type* iteratorStart, const uint32_t capacity)
        : iterator(iteratorStart), iteratorEnd(iteratorStart + capacity)
    {}

    [[nodiscard]] constexpr bool writeAndAdvance(const Type& value)
    {
        if (iterator < iteratorEnd)
        {
            *iterator = value;
            iterator++;
            return true;
        }
        return false;
    }
};

/// @brief A minimal ASCII StringView with shortened name to be used in TypeToString.
struct Sv
{
    const char* data;   ///< Pointer to the start of ASCII string
    uint32_t    length; ///< Number of bytes of the ASCII string

    /// @brief  Construct empty Sv
    constexpr Sv() : data(nullptr), length(0) {}

    /// @brief Construct Sv from a pointer to a char* and a length
    /// @param data Pointer to the string
    /// @param length Number of bytes representing length of the string
    constexpr Sv(const char* data, uint32_t length) : data(data), length(length) {}

    /// @brief Construct Sv from a string literal
    /// @tparam N Number of characters in the string
    /// @param data Pointer to the array of characters
    template <uint32_t N>
    constexpr Sv(const char (&data)[N]) : data(data), length(N - 1)
    {}

    constexpr bool operator==(const Sv other) const
    {
        if (length != other.length)
            return false;
        for (uint32_t idx = 0; idx < length; ++idx)
        {
            if (data[idx] != other.data[idx])
                return false;
        }
        return true;
    }
};

/// @brief Returns name of type T (ClNm stands for ClassName, but we shorten it to save bytes on symbol mangling)
/// @tparam T Type to get the name of
/// @return A Sv (StringView) with the given name (works on CLANG, GCC, MSVC)
template <typename T>
static constexpr Sv ClNm()
{
    // clang-format off
#if SC_COMPILER_CLANG || SC_COMPILER_GCC
    const char* name = __PRETTY_FUNCTION__;
    constexpr char separating_char = '=';
    constexpr uint32_t  skip_chars = 2;
    constexpr uint32_t  trim_chars = 1;
    uint32_t         length = 0;
    const char* it = name;
    while (*it != separating_char)
        it++;
    it += skip_chars;
    while (it[length] != 0)
        length++;
    return Sv(it, length - trim_chars);
#else
    const char* name = __FUNCSIG__;
    constexpr char separating_char = '<';
    constexpr char ending_char = '>';
    const char* it = name;
    while (*it != separating_char)
        it++;
    auto itStart = it + 1;
    while (*it != ending_char)
    {
        if (*it == ' ')
            itStart = it + 1;
        it++;
    }
    return Sv(itStart, static_cast<int>(it - itStart));
#endif
    // clang-format on
}

using TypeStringView = Sv;

/// @brief Strips down class name produced by ClNm to reduce binary size (from C++17 going forward)
/// @tparam T A type that will be converted to string
template <typename T>
struct TypeToString
{
#if SC_LANGUAGE_CPP_AT_LEAST_17
  private:
    // In C++ 17 we trim the long string producted by ClassName<T> to reduce executable size
    [[nodiscard]] static constexpr auto TrimClassName()
    {
        constexpr auto className = ClNm<T>();

        ArrayWithSize<char, className.length> trimmedName;
        for (uint32_t i = 0; i < className.length; ++i)
        {
            trimmedName.values[i] = className.data[i];
        }
        trimmedName.size = className.length;
        return trimmedName;
    }

    // Inline static constexpr requires C++17
    static inline constexpr auto value = TrimClassName();

  public:
    [[nodiscard]] static constexpr TypeStringView get() { return TypeStringView(value.values, value.size); }
#else
    [[nodiscard]] static constexpr TypeStringView get()
    {
        auto className = ClNm<T>();
        return TypeStringView(className.data, className.length);
    }
#endif
};

//! @}

} // namespace Reflection
} // namespace SC


namespace SC
{
/// @brief Describe C++ types at compile time for serialization (see @ref library_reflection).
namespace Reflection
{
//! @defgroup group_reflection Reflection
//! @copybrief library_reflection (see @ref library_reflection for more details)
///
/// Reflection generates compile time information of fields in a structure or class. @n
/// Typically this library is used with one of the serialization libraries
/// ( @ref library_serialization_binary or @ref library_serialization_text).
///
/// @note Reflection uses more complex C++ constructs compared to other libraries in this repository.
/// To limit the issue, effort has been spent trying not to use obscure C++ meta-programming techniques.
/// The library uses only template partial specialization and `constexpr`.

//! @addtogroup group_reflection
//! @{

/// @brief Enumeration of possible category types recognized by Reflection.
/// @note We can use only 7 of the 8 bits here, as in TypeInfo we're stealing 1 bit for TypeInfo::hasLink
//! [reflectionSnippet3]
enum class TypeCategory : uint8_t
{
    TypeInvalid = 0, ///< Invalid type sentinel

    // Primitive types
    TypeBOOL     = 1,  ///< Type is `bool`
    TypeUINT8    = 2,  ///< Type is `uint8_t`
    TypeUINT16   = 3,  ///< Type is `uint16_t`
    TypeUINT32   = 4,  ///< Type is `uint32_t`
    TypeUINT64   = 5,  ///< Type is `uint64_t`
    TypeINT8     = 6,  ///< Type is `int8_t`
    TypeINT16    = 7,  ///< Type is `int16_t`
    TypeINT32    = 8,  ///< Type is `int32_t`
    TypeINT64    = 9,  ///< Type is `int64_t`
    TypeFLOAT32  = 10, ///< Type is `float`
    TypeDOUBLE64 = 11, ///< Type is `double`

    // Non primitive types
    TypeStruct = 12, ///< Type is a struct type
    TypeArray  = 13, ///< Type is an array type
    TypeVector = 14, ///< Type is a vector type
};
//! [reflectionSnippet3]

/// @brief A single 8 bytes union holding most important information about a reflected type
/// This structure is expected to be stored in an flat array.
/// Children items are always following parent items in the flat array.
/// For example when a struct is defined, its members are defined as its children.
/// When one of these children is simple primitive type, this type is stored inline with the type itself.
/// When instead a complex type is needed, a linkID is provided.
/// Such link is as offset in the flat array where detailed definition of the complex type exists.
//! [reflectionSnippet4]
struct TypeInfo
{
    bool         hasLink : 1; ///< Contains a link to another type
    TypeCategory type    : 7; ///< Type of typeinfo
    union
    {
        uint8_t numberOfChildren; ///< Only valid when TypeInfo::hasLink == `false`
        uint8_t linkIndex;        ///< Only valid when TypeInfo::hasLink == `true`
    };
    uint16_t sizeInBytes; ///< Size in bytes of the described type

    /// @brief Holds no extended type info
    struct EmptyInfo
    {
    };

    /// @brief Holds extended type info for members of struct
    struct MemberInfo
    {
        uint16_t memberTag;     ///< Used for versioned serialization
        uint16_t offsetInBytes; ///< Used for signature uniqueness and by SerializationBinaryTypeErased
        constexpr MemberInfo(uint8_t memberTag, uint16_t offsetInBytes)
            : memberTag(memberTag), offsetInBytes(offsetInBytes)
        {}
    };

    /// @brief Holds extended type info for structs
    struct StructInfo
    {
        bool isPacked : 1; ///< Ensures no padding (recursively) for the entire span of the struct
        constexpr StructInfo(bool isPacked) : isPacked(isPacked) {}
    };

    /// @brief Holds extended type info for array-like types
    struct ArrayInfo
    {
        uint32_t isPacked    : 1;  ///< Ensures no padding (recursively) for the entire span of the struct
        uint32_t numElements : 31; ///< Number of elements in the array
        constexpr ArrayInfo(bool isPacked, uint32_t numElements) : isPacked(isPacked), numElements(numElements) {}
    };
    union
    {
        EmptyInfo  emptyInfo;
        MemberInfo memberInfo;
        StructInfo structInfo;
        ArrayInfo  arrayInfo;
    };
    //! [reflectionSnippet4]

    /// @brief Constructs an invalid type info.
    constexpr TypeInfo()
        : hasLink(false), type(TypeCategory::TypeInvalid), numberOfChildren(0), sizeInBytes(0), emptyInfo()
    {
        static_assert(sizeof(TypeInfo) == 8, "Size must be 8 bytes");
    }

    /// @brief Constructs a TypeInfo used by Struct Types
    constexpr TypeInfo(TypeCategory type, uint16_t sizeInBytes, StructInfo structInfo)
        : hasLink(false), type(type), numberOfChildren(0), sizeInBytes(sizeInBytes), structInfo(structInfo)
    {}

    /// @brief Constructs a TypeInfo used by Struct Members (children of Struct Type)
    constexpr TypeInfo(TypeCategory type, uint16_t sizeInBytes, MemberInfo member)
        : hasLink(true), type(type), linkIndex(0), sizeInBytes(sizeInBytes), memberInfo(member)
    {}

    /// @brief Constructs a TypeInfo used  by Array-like Types (T[N], Array<T, N> and Vector<T>)
    constexpr TypeInfo(TypeCategory type, uint16_t sizeInBytes, uint8_t numberOfChildren, ArrayInfo arrayInfo)
        : hasLink(false), type(type), numberOfChildren(numberOfChildren), sizeInBytes(sizeInBytes), arrayInfo(arrayInfo)
    {}

    /// @brief Constructs a TypeInfo of given type and size
    constexpr TypeInfo(TypeCategory type, uint16_t sizeInBytes)
        : hasLink(true), type(type), linkIndex(0), sizeInBytes(sizeInBytes), emptyInfo()
    {}

    /// @brief Get number of children (if any) of this info. Only valid when `hasLink` == `false`
    [[nodiscard]] constexpr auto getNumberOfChildren() const { return numberOfChildren; }

    /// @brief Sets the number of children of this typeinfo
    /// @param numChildren New number of children
    /// @return `true` if it has been changed successfully
    [[nodiscard]] constexpr bool setNumberOfChildren(size_t numChildren)
    {
        if (numChildren > static_cast<decltype(numberOfChildren)>(~0ull)) // MaxValue<uint8_t>
            return false;
        numberOfChildren = static_cast<decltype(numberOfChildren)>(numChildren);
        return true;
    }

    /// @brief Check if this type info has a valid link index
    [[nodiscard]] constexpr bool hasValidLinkIndex() const { return hasLink and linkIndex > 0; }

    /// @brief Check if this type info needs to be linked
    [[nodiscard]] constexpr bool needsLinking() const { return hasLink and linkIndex == 0; }

    /// @brief Obtains link valid index (assuming `hasLink` == `true` and needsLinking() == `false`)
    /// @return The link index for this type that indicates the linked type
    [[nodiscard]] constexpr auto getLinkIndex() const { return linkIndex; }

    /// @brief Change Link index for this type
    /// @param newLinkIndex New Link Index
    /// @return `true` if the link index has been changed successfully
    [[nodiscard]] constexpr bool setLinkIndex(ssize_t newLinkIndex)
    {
        if (newLinkIndex > static_cast<decltype(linkIndex)>(~0ull))
            return false;
        linkIndex = static_cast<decltype(linkIndex)>(newLinkIndex);
        return true;
    }

    /// @brief Check if type is primitive
    [[nodiscard]] constexpr bool isPrimitiveType() const { return isPrimitiveCategory(type); }

    /// @brief Check if type is primitive or it's a struct with `isPacked` property == `true`
    [[nodiscard]] constexpr bool isPrimitiveOrPackedStruct() const
    {
        if (isPrimitiveType())
            return true;
        return (type == Reflection::TypeCategory::TypeStruct) and structInfo.isPacked;
    }

    [[nodiscard]] static constexpr bool isPrimitiveCategory(TypeCategory category)
    {
        return category >= TypeCategory::TypeBOOL and category <= TypeCategory::TypeDOUBLE64;
    }
};
//! @}

/// @brief Basic class template that must be partially specialized for each type.
/// @tparam T The type to specialize for
template <typename T>
struct Reflect;

/// @brief Class template used to check if a given type `IsPacked` property is `true` at compile time
template <typename T, typename SFINAESelector = void>
struct ExtendedTypeInfo;

//-----------------------------------------------------------------------------------------------------------
// Primitive Types Support
//-----------------------------------------------------------------------------------------------------------

/// @brief Base struct for all primitive types
struct ReflectPrimitive
{
    template <typename TypeVisitor>
    [[nodiscard]] static constexpr bool build(TypeVisitor&)
    {
        return true;
    }
};

// clang-format off
template <> struct Reflect<char>     : public ReflectPrimitive {static constexpr auto getCategory(){return TypeCategory::TypeINT8;}};
template <> struct Reflect<uint8_t>  : public ReflectPrimitive {static constexpr auto getCategory(){return TypeCategory::TypeUINT8;}};
template <> struct Reflect<uint16_t> : public ReflectPrimitive {static constexpr auto getCategory(){return TypeCategory::TypeUINT16;}};
template <> struct Reflect<uint32_t> : public ReflectPrimitive {static constexpr auto getCategory(){return TypeCategory::TypeUINT32;}};
template <> struct Reflect<uint64_t> : public ReflectPrimitive {static constexpr auto getCategory(){return TypeCategory::TypeUINT64;}};
template <> struct Reflect<int8_t>   : public ReflectPrimitive {static constexpr auto getCategory(){return TypeCategory::TypeINT8;}};
template <> struct Reflect<int16_t>  : public ReflectPrimitive {static constexpr auto getCategory(){return TypeCategory::TypeINT16;}};
template <> struct Reflect<int32_t>  : public ReflectPrimitive {static constexpr auto getCategory(){return TypeCategory::TypeINT32;}};
template <> struct Reflect<int64_t>  : public ReflectPrimitive {static constexpr auto getCategory(){return TypeCategory::TypeINT64;}};
template <> struct Reflect<float>    : public ReflectPrimitive {static constexpr auto getCategory(){return TypeCategory::TypeFLOAT32;}};
template <> struct Reflect<double>   : public ReflectPrimitive {static constexpr auto getCategory(){return TypeCategory::TypeDOUBLE64;}};
template <> struct Reflect<bool>     : public ReflectPrimitive {static constexpr auto getCategory(){return TypeCategory::TypeBOOL;}};

/// @brief Checks if a given type is primitive
template <typename T> struct IsPrimitive { static constexpr bool value = TypeInfo::isPrimitiveCategory(Reflect<T>::getCategory()); };
template <typename T> struct IsStruct { static constexpr bool value = Reflect<T>::getCategory() == TypeCategory::TypeStruct; };

// clang-format on

template <typename T>
struct ExtendedTypeInfo<T, typename SC::TypeTraits::EnableIf<IsPrimitive<T>::value>::type>
{
    // Primitive types are packed
    static constexpr bool IsPacked = true;
};

//-----------------------------------------------------------------------------------------------------------
// Arrays Support
//-----------------------------------------------------------------------------------------------------------
template <typename T, size_t N>
struct Reflect<T[N]>
{
    static constexpr TypeCategory getCategory() { return TypeCategory::TypeArray; }

    template <typename TypeVisitor>
    [[nodiscard]] static constexpr bool build(TypeVisitor& builder)
    {
        using Type = typename TypeVisitor::Type;

        // Add array type
        constexpr bool isPacked = ExtendedTypeInfo<T>::IsPacked;
        if (not builder.addType(Type::template createArray<T[N]>("Array", 1, TypeInfo::ArrayInfo{isPacked, N})))
            return false;

        // Add dependent item type
        if (not builder.addType(Type::template createGeneric<T>()))
            return false;

        return true;
    }
};

template <typename T, int N>
struct ExtendedTypeInfo<T[N]>
{
    // Arrays are packed if T is packed
    static constexpr bool IsPacked = ExtendedTypeInfo<T>::IsPacked;
};

//-----------------------------------------------------------------------------------------------------------
// Structs Support
//-----------------------------------------------------------------------------------------------------------
template <typename Type>
struct ReflectStruct
{
    using T = Type;

    [[nodiscard]] static constexpr TypeCategory getCategory() { return TypeCategory::TypeStruct; }

    template <typename TypeVisitor>
    [[nodiscard]] static constexpr bool build(TypeVisitor& builder)
    {
        // Add struct type
        if (not builder.addType(TypeVisitor::Type::template createStruct<T>()))
            return false;

        // Add all struct member
        if (not Reflect<Type>::visit(builder))
            return false;
        return true;
    }
};

/// @brief Visit all struct members to gather sum of their sizes (helper for IsPacked).
template <typename T>
struct ExtendedStructTypeInfo
{
    size_t memberSizeSum = 0;
    bool   IsPacked      = false;

    constexpr ExtendedStructTypeInfo()
    {
        // Let's call operator() for each member
        if (Reflect<T>::visit(*this))
        {
            // Structs are packed if all of its members are packed (checked in `operator()`)
            // and in addition to that, summed size of all members is `==` `sizeof(T)`
            IsPacked = memberSizeSum == sizeof(T);
        }
    }

    template <typename R, int N>
    constexpr bool operator()(int memberTag, R T::* member, const char (&name)[N], size_t offset)
    {
        (void)(memberTag);
        (void)(name);
        (void)(member);
        (void)(offset);
        if (not ExtendedTypeInfo<R>().IsPacked)
        {
            return false; // If a given type is not packed, let's stop iterating
        }
        memberSizeSum += sizeof(R);
        return true; // continue iterating
    }
};

template <typename T, typename SFINAESelector>
struct ExtendedTypeInfo
{
    // Struct is packed if all of its members are packed and sum of their size equals size of struct.
    static constexpr bool IsPacked = ExtendedStructTypeInfo<T>().IsPacked;
};

template <typename MemberVisitor, typename Container, typename ItemType, int N>
struct VectorArrayVTable
{
    static constexpr bool build(MemberVisitor&) { return true; }
};
} // namespace Reflection
} // namespace SC

// Handy Macros to avoid some typing when wrapping structs

/// @brief Implement a `Reflect<StructName>` typing less text
#define SC_REFLECT_STRUCT_VISIT(StructName)                                                                            \
    template <>                                                                                                        \
    struct SC::Reflection::Reflect<StructName> : SC::Reflection::ReflectStruct<StructName>                             \
    {                                                                                                                  \
        template <typename TypeVisitor>                                                                                \
        static constexpr bool visit(TypeVisitor&& builder)                                                             \
        {                                                                                                              \
            SC_COMPILER_WARNING_PUSH_OFFSETOF                                                                          \
            return true

/// @brief Reflect a single struct `MEMBER`, giving it an `MEMBER_TAG` integer. Can exist after
/// `SC_REFLECT_STRUCT_VISIT`.
#define SC_REFLECT_STRUCT_FIELD(MEMBER_TAG, MEMBER)                                                                    \
    and builder(MEMBER_TAG, &T::MEMBER, #MEMBER, SC_COMPILER_OFFSETOF(T, MEMBER))

/// @brief Closes `Reflect<StructName>`struct opened by `SC_REFLECT_STRUCT_VISIT`
#define SC_REFLECT_STRUCT_LEAVE()                                                                                      \
    ;                                                                                                                  \
    SC_COMPILER_WARNING_POP_OFFSETOF                                                                                   \
    }                                                                                                                  \
    }                                                                                                                  \
    ;

//----------------------------------------------------------------------------------------------------------------------
// Reflection/ReflectionSchemaCompiler.h
//----------------------------------------------------------------------------------------------------------------------
// Copyright (c) Stefano Cristiano
// SPDX-License-Identifier: MIT

namespace SC
{
namespace Reflection
{
//! @addtogroup group_reflection
//! @{

/// @brief Creates a schema linking a series of SchemaType
/// @tparam SchemaBuilder The builder used to obtain Virtual Tables
template <typename SchemaBuilder>
struct SchemaCompiler
{
  private:
    using Type              = typename SchemaBuilder::Type;
    using TypeBuildFunction = typename Type::TypeBuildFunction;
    using VirtualTablesType = decltype(SchemaBuilder::vtables);

    /// @brief Holds results as a maxed out array of size `MAX_TOTAL_TYPES`
    template <uint32_t MAX_TOTAL_TYPES>
    struct FlatFullResult
    {
        ArrayWithSize<Type, MAX_TOTAL_TYPES> types;
        VirtualTablesType                    vtables;
    };

    /// @brief Holds only the actual `NUM_TYPES` `<=` `MAX_TOTAL_TYPES` reducing executable size.
    template <uint32_t NUM_TYPES>
    struct FlatTrimmedResult
    {
        ArrayWithSize<TypeInfo, NUM_TYPES>       typeInfos;
        ArrayWithSize<TypeStringView, NUM_TYPES> typeNames;
        VirtualTablesType                        vtables;
    };
    template <typename Iterator, typename BinaryPredicate>
    static constexpr void bubbleSort(Iterator first, Iterator last, BinaryPredicate predicate)
    {
        if (first >= last)
        {
            return;
        }
        bool doSwap = true;
        while (doSwap)
        {
            doSwap      = false;
            Iterator p0 = first;
            Iterator p1 = first + 1;
            while (p1 != last)
            {
                if (predicate(*p1, *p0))
                {
                    swap(*p1, *p0);
                    doSwap = true;
                }
                ++p0;
                ++p1;
            }
        }
    }
    template <uint32_t MAX_TYPES>
    [[nodiscard]] static constexpr bool appendTypesTo(ArrayWithSize<Type, MAX_TYPES>& types, TypeBuildFunction build,
                                                      SchemaBuilder& builder)
    {
        // Let builder write to a slice of our available space in types array
        const auto baseLinkID = types.size;
        builder.currentLinkID = types.size;
        builder.types         = {types.values + types.size, MAX_TYPES - types.size};

        if (build(builder))
        {
            // Set number of children for parent type and update types array size
            const auto numberOfTypes    = builder.currentLinkID - baseLinkID;
            const auto numberOfChildren = numberOfTypes - 1;
            if (numberOfChildren > static_cast<decltype(TypeInfo::numberOfChildren)>(~0ull))
                return false;
            if (not types.values[baseLinkID].typeInfo.setNumberOfChildren(numberOfChildren))
                return false;

            struct OrderByMemberOffset
            {
                constexpr bool operator()(const Type& a, const Type& b) const
                {
                    return a.typeInfo.memberInfo.offsetInBytes < b.typeInfo.memberInfo.offsetInBytes;
                }
            };
            if (types.values[baseLinkID].typeInfo.type == TypeCategory::TypeStruct and
                types.values[baseLinkID].typeInfo.structInfo.isPacked)
            {
                // This is a little help for Binary Serialization, as packed structs end up serialized as is
                bubbleSort(types.values + baseLinkID + 1, types.values + baseLinkID + 1 + numberOfChildren,
                           OrderByMemberOffset());
            }
            types.size += numberOfTypes;
            return true;
        }
        return false;
    }

    template <uint32_t MAX_LINK_BUFFER_SIZE, uint32_t MAX_TOTAL_TYPES, typename Func>
    constexpr static FlatFullResult<MAX_TOTAL_TYPES> compileAllTypesFor(Func func)
    {
        // Collect all types
        FlatFullResult<MAX_TOTAL_TYPES> result;

        SchemaBuilder container(result.types.values, MAX_TOTAL_TYPES);

#if SC_COMPILER_GCC // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=77911
        ArrayWithSize<TypeStringView, MAX_LINK_BUFFER_SIZE> alreadyVisitedTypes;
#else
        ArrayWithSize<TypeBuildFunction, MAX_LINK_BUFFER_SIZE> alreadyVisitedTypes;
#endif
        ArrayWithSize<uint32_t, MAX_LINK_BUFFER_SIZE> alreadyVisitedLinkID;
        if (not appendTypesTo(result.types, func, container))
        {
            return {};
        }

        // Link all collected types
        uint32_t typeIndex = 1;
        while (typeIndex < result.types.size)
        {
            Type& type = result.types.values[typeIndex];
            if (not type.typeInfo.isPrimitiveType() and type.typeInfo.needsLinking())
            {
                uint32_t outIndex = 0;
#if SC_COMPILER_GCC // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=77911
                if (alreadyVisitedTypes.contains(type.typeName, &outIndex))
#else
                if (alreadyVisitedTypes.contains(type.typeBuild, &outIndex))
#endif
                {
                    if (not type.typeInfo.setLinkIndex(alreadyVisitedLinkID.values[outIndex]))
                        return {};
                }
                else
                {
                    if (not type.typeInfo.setLinkIndex(result.types.size))
                        return {};
                    if (not alreadyVisitedLinkID.push_back(result.types.size))
                        return {};
#if SC_COMPILER_GCC // https://gcc.gnu.org/bugzilla/show_bug.cgi?id=77911
                    if (not alreadyVisitedTypes.push_back(type.typeName))
#else
                    if (not alreadyVisitedTypes.push_back(type.typeBuild))
#endif
                        return {};
                    if (not appendTypesTo(result.types, type.typeBuild, container))
                        return {};
                }
            }
            typeIndex++;
        }
        result.vtables = container.vtables;
        return result;
    }

  public:
    /// @brief Returns a `constexpr` compiled trimmed flat schema for type `T`.
    /// @tparam T The type to be compiled
    /// @tparam MAX_LINK_BUFFER_SIZE Maximum number of "complex types" (anything that is not a primitive) that can be
    /// built
    /// @tparam MAX_TOTAL_TYPES Maximum number of types (struct members). When using constexpr it will trim it to actual
    /// size.
    /// @returns FlatTrimmedResult with only the required compiled types.
    template <typename T, uint32_t MAX_LINK_BUFFER_SIZE = 20, uint32_t MAX_TOTAL_TYPES = 100>
    static constexpr auto compile()
    {
        constexpr auto schema =
            compileAllTypesFor<MAX_LINK_BUFFER_SIZE, MAX_TOTAL_TYPES>(&Reflect<T>::template build<SchemaBuilder>);
        static_assert(schema.types.size > 0, "Something failed in compileAllTypesFor");

        // Trim the returned FlatTrimmedResult only to the effective number of types
        FlatTrimmedResult<schema.types.size> result;
        for (uint32_t i = 0; i < schema.types.size; ++i)
        {
            result.typeInfos.values[i] = schema.types.values[i].typeInfo;
            result.typeNames.values[i] = schema.types.values[i].typeName;
        }
        result.typeInfos.size = schema.types.size;
        result.typeNames.size = schema.types.size;
        result.vtables        = schema.vtables;
        return result;
    }
};

/// @brief Holds together a TypeInfo, a StringView and a type-erased builder function pointer
/// @tparam TypeVisitor The type of member visitor that is parameter of the builder function
template <typename TypeVisitor>
struct SchemaType
{
    using TypeBuildFunction = bool (*)(TypeVisitor& builder);

    TypeInfo          typeInfo;
    TypeStringView    typeName;
    TypeBuildFunction typeBuild;

    constexpr SchemaType() : typeBuild(nullptr) {}
    constexpr SchemaType(const TypeInfo typeInfo, TypeStringView typeName, TypeBuildFunction typeBuild)
        : typeInfo(typeInfo), typeName(typeName), typeBuild(typeBuild)
    {}

    /// @brief Create from a generic type T
    template <typename T>
    [[nodiscard]] static constexpr SchemaType createGeneric()
    {
        return {TypeInfo(Reflect<T>::getCategory(), sizeof(T)), TypeToString<T>::get(), &Reflect<T>::build};
    }

    /// @brief Create from a Struct type T
    template <typename T>
    [[nodiscard]] static constexpr SchemaType createStruct(TypeStringView name = TypeToString<T>::get())
    {
        TypeInfo::StructInfo structInfo(ExtendedTypeInfo<T>::IsPacked);
        return {TypeInfo(Reflect<T>::getCategory(), sizeof(T), structInfo), name, &Reflect<T>::build};
    }

    /// @brief Create from a struct member with given name, memberTag and offset
    template <typename R, typename T, int N>
    [[nodiscard]] static constexpr SchemaType createMember(uint8_t memberTag, R T::*, const char (&name)[N],
                                                           size_t offset)
    {
        const auto info = TypeInfo::MemberInfo(memberTag, static_cast<SC::uint16_t>(offset));
        return {TypeInfo(Reflect<R>::getCategory(), sizeof(R), info), TypeStringView(name, N - 1), &Reflect<R>::build};
    }

    /// @brief Create from an array-like type
    template <typename T>
    [[nodiscard]] static constexpr SchemaType createArray(TypeStringView name, uint8_t numChildren,
                                                          TypeInfo::ArrayInfo arrayInfo)
    {
        return {TypeInfo(Reflect<T>::getCategory(), sizeof(T), numChildren, arrayInfo), name, &Reflect<T>::build};
    }
};

/// @brief Common code for derived class to create a SchemaBuilder suitable for SC::Reflection::SchemaCompiler
template <typename TypeVisitor>
struct SchemaBuilder
{
    using Type = SchemaType<TypeVisitor>;

    uint32_t currentLinkID;

    WritableRange<Type> types;

    constexpr SchemaBuilder(Type* output, const uint32_t capacity) : currentLinkID(0), types(output, capacity) {}

    template <typename R, typename T, int N>
    [[nodiscard]] constexpr bool operator()(uint8_t memberTag, R T::* field, const char (&name)[N], size_t offset)
    {
        currentLinkID++;
        return types.writeAndAdvance(Type::createMember(memberTag, field, name, offset));
    }

    [[nodiscard]] constexpr bool addType(Type type)
    {
        currentLinkID++;
        return types.writeAndAdvance(type);
    }
};

/// @brief A schema builder that doesn't build any virtual table
struct FlatSchemaBuilder : public SchemaBuilder<FlatSchemaBuilder>
{
    struct EmptyVTables
    {
    };
    EmptyVTables vtables;
    constexpr FlatSchemaBuilder(Type* output, const uint32_t capacity) : SchemaBuilder(output, capacity) {}
};

/// @brief Default schema not building any virtual table
using Schema = Reflection::SchemaCompiler<FlatSchemaBuilder>;
//! @}

} // namespace Reflection

} // namespace SC


#endif // SANE_CPP_REFLECTION_HEADER
