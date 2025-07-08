#include <cerrno>
#include <cstring>
#include <fstream>
#include <mlc/Kernel.h>
#include <stdexcept>
#include <string>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

mini_jit::Kernel::~Kernel() noexcept
{
    release_memory();
}

void mini_jit::Kernel::add_instr(uint32_t ins)
{
    m_buffer.push_back(ins);

    // update labels
    for (auto& label : m_labels)
    {
        label.second += 1;
    }
}

void mini_jit::Kernel::add_instr(std::vector<uint32_t> ins)
{
    m_buffer.reserve(m_buffer.size() + ins.size());
    m_buffer.insert(m_buffer.end(), ins.begin(), ins.end());

    // update labels
    for (auto& label : m_labels)
    {
        label.second += ins.size();
    }
}

void mini_jit::Kernel::add_label(std::string const& label)
{
    if (m_labels.find(label) != m_labels.end())
    {
        throw std::runtime_error("Label already exists: " + label);
    }
    m_labels[label] = 0;
}

int mini_jit::Kernel::getInstrCountFromLabel(std::string const& label) const
{
    auto it = m_labels.find(label);
    if (it == m_labels.end())
    {
        throw std::runtime_error("Label not found: " + label);
    }
    return it->second;
}

std::size_t mini_jit::Kernel::get_size() const
{
    return m_buffer.size() * 4;
}

void* mini_jit::Kernel::alloc_mmap(std::size_t num_bytes) const
{
    void* l_mem = mmap(0,
                       num_bytes,
                       PROT_READ | PROT_WRITE,
                       MAP_PRIVATE | MAP_ANONYMOUS,
                       -1,
                       0);

    if (l_mem == MAP_FAILED)
    {
        throw std::runtime_error("Failed to allocate memory: " + std::string(std::strerror(errno)));
    }

    return l_mem;
}

void mini_jit::Kernel::release_mmap(std::size_t num_bytes,
                                    void*       mem) const
{
    int l_res = munmap(mem,
                       num_bytes);

    if (l_res == -1)
    {
        throw std::runtime_error("Failed to release memory");
    }
}

void mini_jit::Kernel::set_exec(std::size_t num_bytes,
                                void*       mem) const
{
    int l_res = mprotect(mem,
                         num_bytes,
                         PROT_READ | PROT_EXEC);

    if (l_res == -1)
    {
        throw std::runtime_error("Failed to set memory executable: " + std::string(std::strerror(errno)));
    }
}

void mini_jit::Kernel::set_kernel()
{
    release_memory();

    if (m_buffer.empty())
    {
        return;
    }

    // alloc kernel memory
    m_size_alloc = m_buffer.size() * 4;
    try
    {
        m_kernel = (void*)alloc_mmap(m_size_alloc);
    }
    catch (std::runtime_error& e)
    {
        throw std::runtime_error("Failed to allocate memory for kernel: " + std::string(e.what()));
    }

    // copy instruction words from buffer to kernel memory
    for (std::size_t l_in = 0; l_in < m_buffer.size(); l_in++)
    {
        reinterpret_cast<uint32_t*>(m_kernel)[l_in] = m_buffer[l_in];
    }

    // clear cache
    char* l_kernel_ptr = reinterpret_cast<char*>(m_kernel);
    __builtin___clear_cache(l_kernel_ptr,
                            l_kernel_ptr + m_buffer.size() * 4);

    // set executable
    set_exec(m_size_alloc,
             m_kernel);
}

void const* mini_jit::Kernel::get_kernel() const
{
    return m_kernel;
}

void mini_jit::Kernel::release_memory()
{
    if (m_kernel != nullptr)
    {
        release_mmap(m_size_alloc,
                     m_kernel);
    }
    m_size_alloc = 0;

    m_kernel = nullptr;
}

void mini_jit::Kernel::create_dir_if_missing(const char* dir) const
{
    struct stat st;
    if (stat(dir, &st) != 0)
    {
        if (mkdir(dir, 0755) != 0)
        {
            if (errno != EEXIST)
            {
                throw std::runtime_error(std::string("Failed to create directory: ") + dir);
            }
        }
    }
    else if (!S_ISDIR(st.st_mode))
    {
        throw std::runtime_error(std::string("\"") + dir + "\" exists but is not a directory");
    }
}

void mini_jit::Kernel::write(char const* filename) const
{
    create_dir_if_missing("build");
    create_dir_if_missing("build/kernels");

    // edit file path
    std::string filepath = std::string("build/kernels/") + std::string(filename);

    std::ofstream l_out(filepath,
                        std::ios::out | std::ios::binary);
    if (!l_out)
    {
        throw std::runtime_error("Failed to open file: " + filepath);
    }

    l_out.write(reinterpret_cast<char const*>(m_buffer.data()),
                m_buffer.size() * 4);
}
