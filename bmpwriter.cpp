#include <iostream>
#include <fstream>
#include <vector>

#pragma pack(push, 1)
struct BMPHeader {
    uint16_t bfType = 0x4D42; // 'BM'
    uint32_t bfSize = 0;
    uint16_t bfReserved1 = 0;
    uint16_t bfReserved2 = 0;
    uint32_t bfOffBits = 54;
};

struct DIBHeader {
    uint32_t biSize = 40;
    int32_t biWidth = 0;
    int32_t biHeight = 0;
    uint16_t biPlanes = 1;
    uint16_t biBitCount = 24;
    uint32_t biCompression = 0;
    uint32_t biSizeImage = 0;
    int32_t biXPelsPerMeter = 0;
    int32_t biYPelsPerMeter = 0;
    uint32_t biClrUsed = 0;
    uint32_t biClrImportant = 0;
};
#pragma pack(pop)

struct color {
    float r, g, b;
};

void saveBMP(const char* filename, int width, int height, const void* rgbDataVoid) {
    const color* rgbData = static_cast<const color*>(rgbDataVoid);

    BMPHeader bmpHeader;
    DIBHeader dibHeader;

    dibHeader.biWidth = width;
    dibHeader.biHeight = -height;
    dibHeader.biSizeImage = ((width * 3 + 3) & ~3) * height; // row size must be multiple of 4 bytes

    bmpHeader.bfSize = sizeof(BMPHeader) + sizeof(DIBHeader) + dibHeader.biSizeImage;

    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        return;
    }

    file.write(reinterpret_cast<const char*>(&bmpHeader), sizeof(bmpHeader));
    file.write(reinterpret_cast<const char*>(&dibHeader), sizeof(dibHeader));

    std::vector<uint8_t> rowBuffer((width * 3 + 3) & ~3); // 4-byte aligned row buffer

    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            const color& col = rgbData[y * width + x];
            rowBuffer[x * 3 + 0] = static_cast<uint8_t>(col.b * 255); // BMP stores BGR format
            rowBuffer[x * 3 + 1] = static_cast<uint8_t>(col.g * 255);
            rowBuffer[x * 3 + 2] = static_cast<uint8_t>(col.r * 255);
        }
        file.write(reinterpret_cast<const char*>(rowBuffer.data()), rowBuffer.size());
    }

    file.close();
}
