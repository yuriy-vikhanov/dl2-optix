#include "PngWriter.h"
#include <Windows.h>
#include <wincodec.h>
#include <wrl/client.h>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace nurbs {
namespace {
void checkHr(HRESULT result,const char* api,const char* file,int line) {
    if(SUCCEEDED(result)) return;
    std::ostringstream message;
    message << "WIC/COM " << api << " failed: HRESULT 0x" << std::hex << static_cast<unsigned long>(result) << " at " << file << ':' << std::dec << line;
    throw std::runtime_error(message.str());
}
#define WIC_CHECK(call) checkHr((call),#call,__FILE__,__LINE__)
struct ComApartment {
    ComApartment() { WIC_CHECK(CoInitializeEx(nullptr,COINIT_MULTITHREADED)); }
    ~ComApartment() { CoUninitialize(); }
};
}
void writePng(const std::filesystem::path& output,unsigned width,unsigned height,std::span<const uchar4> pixels) {
    static_assert(sizeof(uchar4)==4);
    const std::uint64_t bytes=static_cast<std::uint64_t>(width)*height*4;
    if(width==0 || height==0 || pixels.size()!=static_cast<std::size_t>(width)*height || bytes>std::numeric_limits<UINT>::max()) throw std::runtime_error("Invalid RGBA image dimensions for WIC");
    if(!output.parent_path().empty()) std::filesystem::create_directories(output.parent_path());
    {
        ComApartment apartment;
        using Microsoft::WRL::ComPtr;
        ComPtr<IWICImagingFactory> factory;
        WIC_CHECK(CoCreateInstance(CLSID_WICImagingFactory,nullptr,CLSCTX_INPROC_SERVER,IID_PPV_ARGS(factory.GetAddressOf())));
        ComPtr<IWICStream> stream; WIC_CHECK(factory->CreateStream(stream.GetAddressOf()));
        WIC_CHECK(stream->InitializeFromFilename(output.c_str(),GENERIC_WRITE));
        ComPtr<IWICBitmapEncoder> encoder;
        WIC_CHECK(factory->CreateEncoder(GUID_ContainerFormatPng,nullptr,encoder.GetAddressOf()));
        WIC_CHECK(encoder->Initialize(stream.Get(),WICBitmapEncoderNoCache));
        ComPtr<IWICBitmapFrameEncode> frame; ComPtr<IPropertyBag2> properties;
        WIC_CHECK(encoder->CreateNewFrame(frame.GetAddressOf(),properties.GetAddressOf()));
        WIC_CHECK(frame->Initialize(properties.Get())); WIC_CHECK(frame->SetSize(width,height));
        WICPixelFormatGUID format=GUID_WICPixelFormat32bppRGBA;
        WIC_CHECK(frame->SetPixelFormat(&format));
        auto* data=reinterpret_cast<BYTE*>(const_cast<uchar4*>(pixels.data()));
        if(IsEqualGUID(format,GUID_WICPixelFormat32bppRGBA)) {
            WIC_CHECK(frame->WritePixels(height,width*4,static_cast<UINT>(bytes),data));
        } else {
            // SetPixelFormat negotiates the encoder's supported format (usually
            // BGRA for Windows PNG); convert from the renderer's RGBA buffer.
            ComPtr<IWICBitmap> bitmap;
            WIC_CHECK(factory->CreateBitmapFromMemory(width,height,GUID_WICPixelFormat32bppRGBA,
                width*4,static_cast<UINT>(bytes),data,bitmap.GetAddressOf()));
            ComPtr<IWICFormatConverter> converter;
            WIC_CHECK(factory->CreateFormatConverter(converter.GetAddressOf()));
            WIC_CHECK(converter->Initialize(bitmap.Get(),format,WICBitmapDitherTypeNone,
                nullptr,0.0,WICBitmapPaletteTypeCustom));
            WIC_CHECK(frame->WriteSource(converter.Get(),nullptr));
        }
        WIC_CHECK(frame->Commit()); WIC_CHECK(encoder->Commit());
    } // Close the stream before inspecting the output.
    if(!std::filesystem::is_regular_file(output) || std::filesystem::file_size(output)==0) throw std::runtime_error("PNG output is missing or empty after WIC encoding");
}
}
