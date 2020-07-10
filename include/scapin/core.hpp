#ifndef __SCAPIN_CORE_HPP_20200701061529__
#define __SCAPIN_CORE_HPP_20200701061529__

#if _WIN32
#define DllExport __declspec(dllexport)
#else
#define DllExport
#endif

template <typename T>
std::size_t dim();

#endif
