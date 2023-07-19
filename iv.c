/* Copyright (c) Dmitry "Leo" Kuznetsov 2021 see LICENSE for details */
#include "quick.h"
#include "stb_image.h"
#include "dense.h"

// TODO: see README.md

begin_c

const char* title = "I.V.";

static image_t image;

static char filename[260]; // $(SolutionDir)..\mandrill-4.2.03.png

static void init(void);

app_t app = {
    .class_name = "iv",
    .init = init,
    .min_width = 640,
    .min_height = 640
};

static void* load_image(const byte* data, int64_t bytes, int32_t* w, int32_t* h,
    int32_t* bpp, int32_t preferred_bytes_per_pixel) {
    void* pixels = stbi_load_from_memory((byte const*)data, (int)bytes, w, h,
        bpp, preferred_bytes_per_pixel);
    return pixels;
}

static void load() {
    int r = 0;
    void* data = null;
    int64_t bytes = 0;
    r = crt.memmap_read(filename, &data, &bytes);
    fatal_if_not_zero(r);
    int w = 0;
    int h = 0;
    int bpp = 0; // bytes (!) per pixel
    void* pixels = load_image(data, bytes, &w, &h, &bpp, 0);
    fatal_if_null(pixels);
    gdi.image_init(&image, w, h, bpp, pixels);
    free(pixels);
    crt.memunmap(data, bytes);
}

static void paint(uic_t* ui) {
    gdi.set_brush(gdi.brush_color);
    gdi.set_brush_color(colors.black);
    gdi.fill(0, 0, ui->w, ui->h);
    if (image.w > 0 && image.h > 0) {
        int x = (ui->w - image.w) / 2;
        int y = (ui->h - image.h) / 2;
        gdi.draw_image(x, y, image.w, image.h, &image);
    }
}

static void init(void) {
    app.title = title;
    app.ui->paint = paint;
//  TODO: load first (depth) image from:
//  app.known_folder(known_folder_pictures)
    if (app.argc > 1) {
        strprintf(filename, "%s", app.argv[1]);
        if (access(filename, 0) == 0) {
            load();
        }
    }
}

end_c
