#pragma once

// disable non-sense warning
#pragma warning(disable: 4100) //unreferenced formal parameter
#pragma warning(disable: 4996) //nonsense "CRT deprecated" MS initiative
#pragma warning(disable: 4512) //unassignable object are by design
#pragma warning(disable: 4345) //POD default initialization

// inliend globals
#define SELECTANY _declspec(selectany) ///< use for inline global data 

// Debug-only expressions
#ifdef _DEBUG
#define DBG(x) x
#else
#define DBG(x)
#endif
#define COMMA , ///< use in macro parameters, where no separator is required

#include <tchar.h>
#include <windows.h>
#include <crtdbg.h>

#include <iostream>
#include <iomanip>

#ifndef MSGOUT
#define MSGOUT std::cout// dbg::out()
#endif

namespace PerseusLib
{ 
namespace Utils
{
namespace dbg{
// \namespace GE_::dbg contains debug helpers

typedef std::basic_ostream<TCHAR> ostream_t; ///< type of stream

/// inner details
namespace report_h{

/// contains globals data and related functions
class globals
{
public:
  ostream_t stream; ///< the debug stream
  /// stream buffer
  class streambuff:
      public std::basic_streambuf<TCHAR>
  {
    static const size_t BUFFSZ = 1;
    TCHAR buffer[BUFFSZ];
  public:
    /// imitialize the debug library
    streambuff()
    {
      _CrtSetDbgFlag(
            _CRTDBG_LEAK_CHECK_DF|
            _CRTDBG_ALLOC_MEM_DF|
            _CRTDBG_CHECK_ALWAYS_DF);
      OutputDebugStringA("creating debug stream buffer for ");
      OutputDebugStringA(typeid(char_type).name());
      OutputDebugStringA("\n");
      pubsetbuf(buffer,BUFFSZ-1);
    }
    /// finalize the debug
    ~streambuff()
    {
      OutputDebugStringA("deleting debug stream buffer for ");
      OutputDebugStringA(typeid(char_type).name());
      OutputDebugStringA("\n");
    }
    /// makes the streambuff working on \c buffer
    virtual basic_streambuf* setbuf(TCHAR* _Buffer, std::streamsize _Count)
    {
      setp(buffer,buffer,buffer+BUFFSZ-1);
      return this;
    }

    /// sync and rewind when full
    virtual int_type overflow(int_type i = traits_type::eof())
    {
      sync();
      *pptr() = traits_type::to_char_type(i);
      pbump(1);
      return traits_type::not_eof(i);
    }
    /// spit the buffer to the debugger
    virtual int sync()
    {
      char_type* p = pbase();
      int sz = (int)(pptr()-p);
      p[sz] = 0;
      OutputDebugString(p);
      pbump(-sz);
      return 0;
    }
  };
  unsigned depth; ///< the curred recursion detpth
private:
  streambuff buff; ///< the streambuff instance
public:
  /// initialize the stream
  globals()
    : stream(&buff), depth(1)
  {}
};

/// the \c globals global instance
inline globals& global() { static globals g; return g; }

}

/// getting the global debugging stream
inline ostream_t& out() { return report_h::global().stream; }

/// RAII enter/exit tracer
/** \details
      increments / decrements the depth
      **/
class trace
{
  const char* name;
  void* addr;
  unsigned& depth() { return report_h::global().depth; }
public:
  trace(const char* name_, void * addr_=0) : name(name_), addr(addr_)
  {
    if(name)
    {
      out() << std::setw(depth()) << '>' << name;
      if(addr) out() << '[' << addr << ']';
      out() << std::endl;
    }
    ++depth();
  }
  ~trace()
  {
    --depth();
    if(name)
    {
      out() << std::setw(depth()) << '<' << name;
      if(addr) out() << '[' << addr << ']';
      out() << std::endl;
    }
  }
  template<class T>
  ostream_t& operator<<(const T& t) { out() << std::setw(depth()) << '.' << t; return out(); }
};


inline ostream_t& depth(ostream_t& s)
{	s << std::setw(report_h::global().depth) << '.'; return s; }

/// RAII ctor/dtor tracker
/**	\details
      won't increment / decrement the depth
      **/
class track
{
protected:
  const char* name;
  void* addr;
  unsigned& depth() { return report_h::global().depth; }
public:
  track(const char* name_, void * addr_=0) : name(name_), addr(addr_)
  {
    out() << std::setw(depth()) << "C:" << name;
    if(addr) out() << '[' << addr << ']';
    out() << std::endl;
  }
  ~track()
  {
    out() << std::setw(depth()) << "D:" << name;
    if(addr) out() << '[' << addr << ']';
    out() << std::endl;
  }
  template<class T>
  ostream_t& operator<<(const T& t) { out() << std::setw(depth()) << '.' << t; return out(); }
};

template<class T>
class trackobj: public track
{
public:
  trackobj() : track(typeid(T).name(),&name)
  {}
};


inline void init_debug_h() { out(); }
}
}
}
