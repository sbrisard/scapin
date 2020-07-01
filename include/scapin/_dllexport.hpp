#ifndef __SCAPIN__DLLEXPORT_HPP_20200701061529__
#define __SCAPIN__DLLEXPORT_HPP_20200701061529__

#if _WIN32
#define DllExport __declspec(dllexport)
#else
#define DllExport
#endif

#endif
