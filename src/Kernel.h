#include <cstdint>
#include <cstddef>
#include <vector>

#ifndef MINI_JIT_KERNEL_H
#define MINI_JIT_KERNEL_H

namespace mini_jit {
  class Kernel;
}

class mini_jit::Kernel {
  private:
    //! high-level code buffer
    std::vector< uint32_t > m_buffer;

    //! size of the kernel
    std::size_t m_size_alloc = 0;

    //! executable kernel
    void * m_kernel = nullptr;

    /**
     * Allocates memory through POSIX mmap.
     *
     * @param num_bytes number of bytes.
     **/
    void * alloc_mmap( std::size_t num_bytes ) const;

    /**
     * Release POSIX mmap allocated memory.
     *
     * @param num_bytes number of bytes.
     * @param mem pointer to memory which is released.
     **/
    void release_mmap( std::size_t   num_bytes,
                       void        * mem ) const;
  
    /**
     * Sets the given memory region executable.
     *
     * @param num_bytes number of bytes.
     * @param mem point to memory.
     **/
    void set_exec( std::size_t   num_bytes,
                   void        * mem ) const;

    /**
     * Release memory of the kernel if allocated.
     **/
    void release_memory();

  public:
    /**
     * Constructor
     **/
    Kernel(){};

    /**
     * Destructor
     **/
    ~Kernel() noexcept;

    Kernel( Kernel const & ) = delete;
    Kernel & operator=( Kernel const & ) = delete;
    Kernel( Kernel && ) noexcept = delete;
    Kernel & operator=( Kernel && ) noexcept = delete;

    /**
     * Adds an instruction to the code buffer.
     *
     * @param ins instruction which is added.
     **/
    void add_instr( uint32_t ins );

    /**
     * Gets the size of the code buffer.
     *
     * @return size of the code buffer in bytes.
     **/
    std::size_t get_size() const;

    /**
     * Sets the kernel based on the code buffer.
     **/
    void set_kernel();

    /**
     * Gets a pointer to the executable kernel.
     **/
    void const * get_kernel() const;

    /**
     * Writes the code buffer to the given file.
     *
     * @param path path to the file.
     **/
    void write( char const * path ) const;
};

#endif