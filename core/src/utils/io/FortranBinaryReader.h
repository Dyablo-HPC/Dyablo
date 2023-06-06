#include <iostream>

namespace dyablo {

namespace FortranBinaryReader_impl{
namespace {

template< typename T >
T endian_swap(const T& v)
{
    constexpr int N = sizeof(T);
    const char* v_in = reinterpret_cast<const char*>(&v);
    T res;
    char* v_out = reinterpret_cast<char*>(&res);
    for( int i=0; i<N; i++ )
    {
        v_out[i] = v_in[N-1-i];
    }
    return res;
}

} // namespace
} // namespace FortranBinaryFileReader_impl

class FortranBinaryReader
{
public:
  inline static bool _byteswap = false;
  using RecordTag_t = uint32_t;

public:
  template < typename T > 
  static void read_record( std::istream& istr, T* buffer, size_t count )
  {
    using namespace FortranBinaryReader_impl;
    // Fortran files write tags at beginning and end of writes containing the number for bytes written
    RecordTag_t begin_tag;
    istr.read((char*)&begin_tag, sizeof(RecordTag_t));
    
    {
      RecordTag_t record_size = _byteswap ? endian_swap(begin_tag) : begin_tag;
      if( record_size != count*sizeof(T) )
      {
        DYABLO_ASSERT_HOST_RELEASE( endian_swap(record_size) == count,
          "Fortran record tag doesn't match required read size.\n"
          "required size : " << count*sizeof(T) << "\n"
          "Begin tag is " << record_size << " ( " << endian_swap(record_size) << " with swapped endianness )" );
        _byteswap = !_byteswap;
      }
    }
    
    istr.read((char*)(buffer), count*sizeof(T));

    if( _byteswap )
    {
        for( int i=0; i<count; i++ )
            buffer[i] = endian_swap(buffer[i]);
    }

    RecordTag_t end_tag;
    istr.read((char*)&end_tag, sizeof(RecordTag_t));
    // Tags are not swapped to match endianness but should still match
    DYABLO_ASSERT_HOST_RELEASE( begin_tag == end_tag, 
          "Fortran record end tag doesn't match begin tag.\n"
          << begin_tag << " != " << end_tag );
    }
};

} // namespace dyablo