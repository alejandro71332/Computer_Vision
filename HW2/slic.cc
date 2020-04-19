//#include <bits/stdc++.h>
#include <cmath>
#include <cstring>
#include <cfloat>
#include <iostream>
#include <algorithm>

using namespace std;

#define STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image.h"
#include "stb_image_write.h"


class Point {
public:
    int x, y;
    int r, g, b;
    Point(int x, int y, int r, int g, int b)
    : x(x), y(y), r(r), g(g), b(b)
    {}
    Point() : Point(0, 0, 0, 0, 0)
    {}
    Point(const Point &p) : x(p.x), y(p.y), r(p.r), g(p.g), b(p.b)
    {}
    Point &operator+=(const Point &o)
    {
        x += o.x;
        y += o.y;
        r += o.r;
        g += o.g;
        b += o.b;
        return *this;
    }
    Point &operator/=(const int total)
    {
        x /= total;
        y /= total;
        r /= total;
        g /= total;
        b /= total;
        return *this;
    }
    Point &operator=(const Point &p)
    {
        if (this == &p)
            return *this;
        x = p.x;
        y = p.y;
        r = p.r;
        g = p.g;
        b = p.b;
        return *this;
    }
    Point operator-(const Point &o) const
    {
        return Point(x - o.x, y - o.y, r - o.r, g - o.g, b - o.b);
    }
    friend bool operator==(const Point &p, const Point &o)
    {
        return p.r == o.r && p.g == o.g && p.b == o.b;
    }
    friend bool operator!=(const Point &p, const Point &o)
    {
        return !(p == o);
    }
    friend ostream &operator<<(ostream &os, const Point &p)
    {
        char buf[256];
        sprintf(buf, "[%u, %u, %u]", p.r, p.g, p.b);
        os << buf;
        return os;
    }
    friend void add(Point &p, const Point &o)
    {
        p.x += o.x;
        p.y += o.y;
        p.r += o.r;
        p.g += o.g;
        p.b += o.b;
    }
    friend void div(Point &p, const int val)
    {
        p.x /= val;
        p.y /= val;
        p.r /= val;
        p.g /= val;
        p.b /= val;
    }
};

class Image {
private:
    unsigned char *data;
    const int width, height;
public:
    Image(unsigned char *data, int width, int height)
    : data(data), width(width), height(height)
    {}
    Image() = delete;
    Point get_point(int x, int y)
    {
        unsigned char *start = &data[y*3*width + x*3];
        unsigned char r = start[0];
        unsigned char g = start[1];
        unsigned char b = start[2];
        return Point(x, y, r, g, b);
    }
};

template <typename T>
T **init_2d(int width, int height)
{
    T **dest = new T*[height];
    for (int i = 0; i < height; ++i)
        dest[i] = new T[width];
    return dest;
}

template <typename T>
void delete_2d(T **src, int height)
{
    for (int i = 0; i < height; ++i)
        delete [] src[i];
    delete [] src;
}

template <typename T>
T **deepcopy_2d(T **src, int width, int height)
{
    T **dest = init_2d<T>(width, height);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            dest[y][x] = src[y][x];
}

double euclidean_distance(Point source, Point dest, int m, int S)
{
    double d_rgb = sqrt(pow(source.r - dest.r, 2) + pow(source.g - dest.g, 2) + pow(source.b - dest.b, 2));
    double d_xy = sqrt(pow((source.x - dest.x)/2.0, 2) + pow((source.y - dest.y)/2.0, 2));
    return sqrt(pow(d_rgb, 2) + pow(d_xy / S, 2) * pow(m, 2));
}

double **transpose(double **src, int width, int height)
{
    double **dest = init_2d<double>(height, width);
    for (int y = 0; y < height; ++y)
        for (int x = 0; x < width; ++x)
            dest[x][y] = src[y][x];
    return dest;
}

double dot_product(const double *A, const double *B, int size) 
{ 
    double product = 0; 
    for (int i = 0; i < size; ++i) 
        product = product + A[i] * B[i]; 
    return product; 
} 

void gradient(unsigned char **src, double **&xDest, double **&yDest, int width, int height)
{
    xDest = new double*[height];
    for (int y = 0; y < height; ++y) {
        double *tmp = new double[width];
        for (int x = 0; x < width; ++x) {
            int left = x - 1;
            int right = x + 1;
            if (left < 0)
                left = 0;
            else if (right == width)
                right = x;

            double grad = (double)(src[y][right] - src[y][left]) / (right - left);
            tmp[x] = grad;
        }
        xDest[y] = tmp;
    }

    double **trans = new double*[width];
    for(int x = 0; x < width; ++x) {
        double *tmp = new double[height];
        for(int y = 0; y < height; ++y) {
            int above = y - 1;
            int below = y + 1;
            if(above < 0)
                above = 0;
            else if(below == height)
                below = y;
            double grad = (double)(src[below][x] - src[above][x]) / (below - above);
            tmp[y] = grad;
        }
        trans[x] = tmp;
    }

    yDest = transpose(trans, height, width);
    delete_2d<double>(trans, width);
}

void remove_disconnected(char **edges, int *, int, int, int);
Point replace_color(Point *, int *, int, int, int, int);



int main(int argc, char **argv)
{
    int img_x, img_y, channels;
    unsigned char *data = stbi_load("wt_slic.png", &img_x, &img_y, &channels, 0);
    if (!data) {
        cerr << "Cannot open file.\n";
        return -1;
    }

    Image *img = new Image(data, img_x, img_y);

    int N = img_y * img_x;
    int K = 150;
    int S = sqrt((double)N / K);
    int m = 200;

    int cluster_width = 50;
    int cluster_height = 50;

    Point *cluster_centers = new Point[K];
    int zz = 0;
    for (int y = S / 2; y < img_y; y += S)
        for (int x = S / 2; x < img_x; x += S, ++zz)
            cluster_centers[zz] = img->get_point(x, y);

    Point *img_pixels = new Point[N];
    for (int y = 0; y < img_y; ++y)
        for (int x = 0; x < img_x; ++x)
            img_pixels[y * img_x + x] = img->get_point(x, y);


    unsigned char **rChan = init_2d<unsigned char>(img_x, img_y);
    unsigned char **gChan = init_2d<unsigned char>(img_x, img_y);
    unsigned char **bChan = init_2d<unsigned char>(img_x, img_y);
    for (int y = 0; y < img_y; ++y) {
        for (int x = 0; x < img_x; ++x) {
            Point p = img->get_point(x, y);
            rChan[y][x] = p.r;
            gChan[y][x] = p.g;
            bChan[y][x] = p.b;
        }
    }

    double **grad_rx, **grad_ry, **grad_gx, **grad_gy, **grad_bx, **grad_by;
    gradient(rChan, grad_rx, grad_ry, img_x, img_y);
    gradient(gChan, grad_gx, grad_gy, img_x, img_y);
    gradient(bChan, grad_bx, grad_by, img_x, img_y);

    double **combined_grad = init_2d<double>(img_x, img_y);

    for(int y = 0; y < img_y; y++){
        for(int x = 0; x < img_x; x++){
            double r_mag = sqrt(pow(grad_rx[y][x], 2) + pow(grad_ry[y][x], 2));
            double g_mag = sqrt(pow(grad_gx[y][x], 2) + pow(grad_gy[y][x], 2));
            double b_mag = sqrt(pow(grad_bx[y][x], 2) + pow(grad_by[y][x], 2));
            combined_grad[y][x] = sqrt(pow(r_mag, 2) + pow(g_mag, 2) + pow(b_mag, 2));
        }
    }

    // deep copy
    Point *cluster_copy = new Point[K];
    memcpy(cluster_copy, cluster_centers, K * sizeof(Point));


    for (int k = 0; k < K; ++k) {
        double min_grad = DBL_MAX;
        Point cluster = cluster_copy[k];
        if (cluster.x == 0 || cluster.y == 0)
            continue;
        for (int y = -1; y < 2; ++y) {
            for (int x = -1; x < 2; ++x) {
                if (combined_grad[cluster_copy[k].y + y][cluster_copy[k].x + x] < min_grad) {
                    Point p = img->get_point(cluster_copy[k].x + x, cluster_copy[k].y + y);
                    cluster_centers[k].r = p.r;
                    cluster_centers[k].g = p.g;
                    cluster_centers[k].b = p.b;
                    cluster_centers[k].x = p.x;
                    cluster_centers[k].y = p.y;
                    min_grad = combined_grad[cluster_copy[k].y + y][cluster_copy[k].x + x];
                }
            }
        }
    }


    double error_threshold = 5;
    double residual_error =  error_threshold + 1;
    int iter = 0;
    int max_iter = 1;

    int *label = new int[N];
    double *distance = new double[N];
    for(int i = 0; i < N; ++i) {
        label[i] = -1;
        distance[i] = DBL_MAX;
    }
            
    while (residual_error > error_threshold) {
        for (int k = 0; k < K; ++k) {
            Point cluster = cluster_centers[k];
            int x_center = cluster.x;
            int y_center = cluster.y;
            for (int y = -S; y <= S; y++) {
                for (int x = -S; x <= S; x++) {
                    if (y + cluster.y < 0 || x + cluster.x < 0 || y + cluster.y >= img_y || x + cluster.x >= img_x)
                        continue;
                    Point pixel_i = img->get_point(cluster.x + x, cluster.y + y);

                    double D = euclidean_distance(cluster, pixel_i, m, S);

                    int loc = pixel_i.y * img_x + pixel_i.x;

                    if (D < distance[loc]) {
                        distance[loc] = D;
                        label[loc] = k;
                    }
                }
            }
        }

        for (int k = 0; k < K; ++k) {
            Point new_center = Point();
            int total_in_cluster = 0;
            for(int j = 0; j < N; ++j) {
                Point pixel = img_pixels[j];
                int x = pixel.x;
                int y = pixel.y;
                if(label[y * img_x + x] == k) {
                    new_center += pixel;
                    total_in_cluster++;
                }
            }
            if (total_in_cluster > 0) {
                new_center /= total_in_cluster;
                Point tmp = cluster_centers[k] - new_center;
                double partial_error[5] = {(double)tmp.x, (double)tmp.y, (double)tmp.r, (double)tmp.g, (double)tmp.b};
                double dot = dot_product(partial_error, partial_error, 5);
                residual_error += sqrt(dot);
                cluster_centers[k] = new_center;
            }
        }
        
        if (++iter > max_iter) {
            residual_error = error_threshold;
        }
    }

    char **edges = init_2d<char>(img_x, img_y);
    for (int i = 0; i < img_y; ++i)
        memset(edges[i], 0, img_x);

    for (int y = 0; y < img_y; ++y) {
        for (int x = 0; x < img_x; ++x) {
            int pos = y * img_x + x;
            int belongs_to = label[pos];
            if (x != 0 && y != 0 && x != img_x - 1 && y != img_y - 1) {
                if (label[pos + 1] != belongs_to || label[(y + 1) * img_x + x] != belongs_to)
                    edges[y][x] = 1;
            }
        }
    }
    remove_disconnected(edges, label, img_x, img_y, 500);


    unsigned char *out = new unsigned char[N * channels];
    Point **color = init_2d<Point>(img_x, img_y);

    for (int y = 0; y < img_y; ++y)
        for (int x = 0; x < img_x; ++x)
            color[y][x] = cluster_centers[label[y * img_x + x]];

    for (int y = 0, px = 0; y < img_y; ++y) {
        for (int x = 0; x < img_x; ++x, px+=3) {
            if (edges[y][x] == 1 || !y || !x || x == img_x - 1 || y == img_y - 1) {
                out[px+0] = 0;
                out[px+1] = 0;
                out[px+2] = 0;
                color[y][x] = Point();
            }
        }
    }
    for (int y = 0; y < img_y; ++y) {
        for (int x = 0; x < img_x; ++x) {
            if (edges[y][x] == -1) {
                Point p = replace_color(cluster_centers, label, x, y, img_x, img_y);
                Point c = cluster_centers[label[y * img_x + x]];
                color[y][x] = p;
            }
        }
    }
    
    for (int y = 0, px = 0; y < img_y; ++y) {
        for (int x = 0; x < img_x; ++x, px += 3) {
            Point p = cluster_centers[label[y * img_x + x]];
            out[px+0] = color[y][x].r;
            out[px+1] = color[y][x].g;
            out[px+2] = color[y][x].b;
        }
    }


    stbi_write_png("slic_output.png", img_x, img_y, 3, out, 0);
    delete img;
    delete [] cluster_centers;
    delete [] img_pixels;
    delete_2d<unsigned char>(rChan, img_y);
    delete_2d<unsigned char>(gChan, img_y);
    delete_2d<unsigned char>(bChan, img_y);
    delete_2d<double>(grad_rx, img_y);
    delete_2d<double>(grad_gx, img_y);
    delete_2d<double>(grad_bx, img_y);
    delete_2d<double>(grad_ry, img_y);
    delete_2d<double>(grad_gy, img_y);
    delete_2d<double>(grad_by, img_y);
    delete_2d<double>(combined_grad, img_y);
    delete [] cluster_copy;
    delete [] label;
    delete [] distance;
    delete_2d<char>(edges, img_y);
    delete_2d<Point>(color, img_y);
    delete [] out;

    stbi_image_free(data);

}

int mostFrequent(int arr[], int n)
{
    sort(arr, arr + n);
    int max = 1, val = *arr, curr = 1;
    for (int i = 1; i < n; i++) {
        if (arr[i] == -1)
            continue;
        if (arr[i] == arr[i - 1]) {
            ++curr;
        } else {
            if (curr > max) {
                max = curr;
                val = arr[i - 1];
            }
            curr = 1;
        }
    }

    if (curr > max)
        val = arr[n - 1];

    return val;
}

Point replace_color(Point *centers, int *label, int x, int y, int width, int height)
{
    const int size = 5;
    int r[size * size] = {0};
    int g[size * size] = {0};
    int b[size * size] = {0};
    int tmp[size * size];
    for (int offy = -size/2; offy <= size/2; ++offy) {
        for (int offx = -size/2; offx <= size/2; ++offx) {
            int yy = y + offy;
            int xx = x + offx;
            if (yy >= 0 && yy < height && xx >= 0 && xx < width) {
                Point p = centers[label[yy * width + xx]];
                r[(offy+size/2) * size + (offx+size/2)] =  p.r;
                g[(offy+size/2) * size + (offx+size/2)] =  p.g;
                b[(offy+size/2) * size + (offx+size/2)] =  p.b;
            } else {
                r[(offy+size/2) * size + (offx+size/2)] = -1;
                g[(offy+size/2) * size + (offx+size/2)] = -1;
                b[(offy+size/2) * size + (offx+size/2)] = -1;
            }
        }
    }
    memcpy(tmp, r, size * sizeof(int));
    int max_freq = mostFrequent(tmp, size);
    int idx;
    for (idx = 0; idx < size; ++idx)
        if (r[idx] == max_freq)
            break;
    return Point(0, 0, r[idx], g[idx], b[idx]);
}

bool **cluster;
int dfs(char **edges, int col, int row, int width, int height, bool **visited)
{
    static int rowNbr[] = { -1, -1, -1, 0, 0, 1, 1, 1 };
    static int colNbr[] = { -1, 0, 1, -1, 1, -1, 0, 1 };
    int count = 0;

    visited[row][col] = true;
    cluster[row][col] = true; // doesn't matter if you're a 0
    for (int k = 0; k < 8; ++k) {
        int y = row + rowNbr[k];
        int x = col + colNbr[k];
        if (y >= 0 && y < height && x >= 0 && x < width && edges[y][x] && !visited[y][x]) {
            count += dfs(edges, x, y, width, height, visited);
        }
    }
    return 1 + count;
}

void remove_disconnected(char **edges, int *label, int width, int height, int threshold)
{
    int i, j, ans, v;
    bool **visited = init_2d<bool>(width, height);
    cluster = init_2d<bool>(width, height);

    for (int i = 0; i < height; ++i) {
        memset(visited[i], 0, width * sizeof(bool));
        memset(cluster[i], 0, width * sizeof(bool));
    }
    
    for (int y = 0; y < height; ++y) {
        for (int x = 0; x < width; ++x) {
            if (edges[y][x] && !visited[y][x]) {
                int count = dfs(edges, x, y, width, height, visited);
                if (count < threshold) {
                    for (int yy = 0; yy < height; ++yy)
                        for (int xx = 0; xx < width; ++xx)
                            if (cluster[yy][xx])
                                edges[yy][xx] = -1;
                }
                for (int i = 0; i < height; ++i)
                    memset(cluster[i], 0, width * sizeof(bool));
            }
        }
    }
    delete_2d<bool>(visited, height);
    delete_2d<bool>(cluster, height);
}
