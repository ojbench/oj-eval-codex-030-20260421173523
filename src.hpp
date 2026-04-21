// Heuristic handwritten digit classifier for 28x28 grayscale images.
// The OJ includes this header and calls judge(img).
// IMAGE_T values are in [0,1], where 1 is white (foreground digit), 0 is black (background).

#pragma once
#include <vector>
#include <queue>
#include <cmath>
#include <algorithm>

typedef std::vector< std::vector<double> > IMAGE_T;

namespace nr_internal {

struct Box { int r0, c0, r1, c1; }; // inclusive bounds

static inline int clampi(int v, int lo, int hi){ return v<lo?lo:(v>hi?hi:v); }

// Binarize with an adaptive threshold based on mean and std.
static std::vector< std::vector<unsigned char> > binarize(const IMAGE_T &img){
    int n = (int)img.size();
    int m = n? (int)img[0].size() : 0;
    double mean = 0.0, m2 = 0.0; int cnt = 0;
    double mn = 1e9, mx = -1e9;
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            double v = img[i][j];
            mean += v; m2 += v*v; cnt++;
            mn = std::min(mn, v); mx = std::max(mx, v);
        }
    }
    mean /= std::max(1, cnt);
    double var = std::max(0.0, m2/std::max(1, cnt) - mean*mean);
    double sd = std::sqrt(var);
    // Foreground is white (1.0), background black (0.0).
    double thr = 0.0;
    if (sd < 1e-6) {
        thr = (mn + mx) * 0.5;
    } else {
        // Pull threshold slightly above mean to reduce noise.
        thr = std::min(1.0, std::max(0.0, mean + 0.25*sd));
        // Ensure not too low if image is very dark
        thr = std::max(thr, 0.35);
    }
    std::vector< std::vector<unsigned char> > bw(n, std::vector<unsigned char>(m, 0));
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            bw[i][j] = (img[i][j] >= thr) ? 1 : 0;
        }
    }
    return bw;
}

static bool any_foreground(const std::vector< std::vector<unsigned char> > &bw){
    for (size_t i=0;i<bw.size();++i){
        for (size_t j=0;j<bw[i].size();++j){
            if (bw[i][j]) return true;
        }
    }
    return false;
}

static Box bounding_box(const std::vector< std::vector<unsigned char> > &bw){
    int n = (int)bw.size(); int m = n? (int)bw[0].size() : 0;
    int r0=n, c0=m, r1=-1, c1=-1;
    for(int i=0;i<n;i++) for(int j=0;j<m;j++) if (bw[i][j]){
        r0 = std::min(r0, i); c0 = std::min(c0, j);
        r1 = std::max(r1, i); c1 = std::max(c1, j);
    }
    if (r1 < r0) { r0=0; c0=0; r1=n-1; c1=m-1; }
    return {r0,c0,r1,c1};
}

static int holes_count_and_centroid(const std::vector< std::vector<unsigned char> > &bw, const Box &b, double &hc_r, double &hc_c, int &holes_area){
    // Count holes in foreground: number of background (0) components fully enclosed by foreground within bbox.
    int n = (int)bw.size(); int m = n? (int)bw[0].size() : 0;
    int h = b.r1 - b.r0 + 1, w = b.c1 - b.c0 + 1;
    if (h <= 0 || w <= 0) { hc_r = hc_c = 0.0; holes_area = 0; return 0; }
    std::vector< std::vector<unsigned char> > vis(h, std::vector<unsigned char>(w, 0));
    std::queue< std::pair<int,int> > q;
    // Mark external background (0) reachable from bbox border in the cropped region
    for(int i=0;i<h;i++){
        for(int j=0;j<w;j++){
            if (i==0||j==0||i==h-1||j==w-1){
                if (!bw[b.r0+i][b.c0+j] && !vis[i][j]){
                    vis[i][j] = 1; q.push({i,j});
                }
            }
        }
    }
    const int dr[4] = {-1,1,0,0};
    const int dc[4] = {0,0,-1,1};
    while(!q.empty()){
        std::pair<int,int> cur = q.front(); q.pop();
        int r = cur.first, c = cur.second;
        for(int k=0;k<4;k++){
            int nr=r+dr[k], nc=c+dc[k];
            if (nr>=0 && nr<h && nc>=0 && nc<w && !vis[nr][nc] && !bw[b.r0+nr][b.c0+nc]){
                vis[nr][nc] = 1; q.push(std::make_pair(nr,nc));
            }
        }
    }
    // Any remaining 0 regions not marked are holes.
    int holes = 0; long long sumr=0, sumc=0; int area=0;
    for(int i=0;i<h;i++){
        for(int j=0;j<w;j++){
            if (!bw[b.r0+i][b.c0+j] && !vis[i][j]){
                holes++;
                std::queue< std::pair<int,int> > qq; qq.push(std::make_pair(i,j)); vis[i][j]=2;
                while(!qq.empty()){
                    std::pair<int,int> cur2 = qq.front(); qq.pop();
                    int r = cur2.first, c = cur2.second;
                    sumr += r; sumc += c; area++;
                    for(int k=0;k<4;k++){
                        int nr=r+dr[k], nc=c+dc[k];
                        if (nr>=0 && nr<h && nc>=0 && nc<w && !bw[b.r0+nr][b.c0+nc] && vis[nr][nc]==0){
                            vis[nr][nc]=2; qq.push(std::make_pair(nr,nc));
                        }
                    }
                }
            }
        }
    }
    holes_area = area;
    if (holes>0 && area>0){ hc_r = (double)sumr/area; hc_c = (double)sumc/area; }
    else { hc_r = hc_c = 0.0; }
    return holes;
}

static void projections(const std::vector< std::vector<unsigned char> > &bw, const Box &b, std::vector<int> &row, std::vector<int> &col){
    int h = b.r1-b.r0+1, w = b.c1-b.c0+1;
    row.assign(h,0); col.assign(w,0);
    for(int i=0;i<h;i++){
        for(int j=0;j<w;j++) if (bw[b.r0+i][b.c0+j]){
            row[i]++; col[j]++;
        }
    }
}

static void zoning(const std::vector< std::vector<unsigned char> > &bw, const Box &b, int gz, std::vector<double> &feat){
    // Simple grid zoning densities (gz x gz)
    int h = b.r1-b.r0+1, w = b.c1-b.c0+1;
    feat.assign(gz*gz, 0.0);
    for(int i=0;i<h;i++){
        for(int j=0;j<w;j++){
            if (!bw[b.r0+i][b.c0+j]) continue;
            int gi = (int)(i * gz / std::max(1,h));
            int gj = (int)(j * gz / std::max(1,w));
            gi = clampi(gi, 0, gz-1); gj = clampi(gj, 0, gz-1);
            feat[gi*gz+gj] += 1.0;
        }
    }
    double norm = 0.0; 
    for(size_t t=0;t<feat.size();++t) norm += feat[t]*feat[t]; 
    norm = std::sqrt(std::max(1e-9, norm));
    for(size_t t=0;t<feat.size();++t) feat[t] /= norm;
}

static int classify(const std::vector< std::vector<unsigned char> > &bw){
    int n = (int)bw.size(); if (n==0) return 0; int m = (int)bw[0].size();
    if (!any_foreground(bw)) return 1; // degenerate, default to 1
    Box b = bounding_box(bw);
    int h = b.r1-b.r0+1, w = b.c1-b.c0+1;

    // Basic geometry
    int mass = 0; long long sumr=0, sumc=0;
    for(int i=b.r0;i<=b.r1;i++) for(int j=b.c0;j<=b.c1;j++) if (bw[i][j]){ mass++; sumr += (i-b.r0); sumc += (j-b.c0); }
    double cr = mass? (double)sumr/mass : h/2.0;
    double cc = mass? (double)sumc/mass : w/2.0;
    double ratio = w / (double)std::max(1,h);

    // Projections
    std::vector<int> prow, pcol; projections(bw, b, prow, pcol);
    int max_row = 0; for(size_t ii=0; ii<prow.size(); ++ii) if (prow[ii] > max_row) max_row = prow[ii];
    int max_col = 0; for(size_t jj=0; jj<pcol.size(); ++jj) if (pcol[jj] > max_col) max_col = pcol[jj];

    // Holes
    double hc_r=0.0, hc_c=0.0; int holes_area=0;
    int holes = holes_count_and_centroid(bw, b, hc_r, hc_c, holes_area);

    // Early rules
    if (holes >= 2) return 8;
    if (holes == 1){
        double hole_y = hc_r / std::max(1, h); // [0,1]
        double hole_x = hc_c / std::max(1, w);
        double hole_frac = holes_area / (double)std::max(1, h*w);
        // 0 tends to be rounder, hole near center, and comparatively large
        if (ratio > 0.75 && ratio < 1.25 && hole_frac > 0.10 && std::fabs(hole_x-0.5) < 0.18 && std::fabs(hole_y-0.5) < 0.18){
            return 0;
        }
        // Distinguish 6 vs 9 by hole position: lower -> 6, upper -> 9
        if (hole_y >= 0.5) return 6; else return 9;
    }

    // No hole: likely 1,2,3,4,5,7
    // 1: very slender and centered
    if (ratio < 0.45 && max_col > std::max(2, h/3)){
        return 1;
    }
    // 7: strong top bar, sparse bottom-left, right diagonal
    int top_q = std::min(h, std::max(2, h/4));
    int top_sum = 0; for(int i=0;i<top_q;i++) top_sum += prow[i];
    int rest_sum = 0; for(int i=top_q;i<h;i++) rest_sum += prow[i];
    if (top_sum > rest_sum && ratio > 0.7){
        return 7;
    }
    // 4: vertical and cross at mid; check mid-row and left column densities
    int midr = h/2; int midc = w/2;
    int midr_sum = (midr>=0 && midr<h) ? prow[midr] : 0;
    int left_col = (0<w) ? pcol[0] : 0;
    if (midr_sum > mass/5 && left_col > mass/6 && ratio < 1.0){
        return 4;
    }
    // Use coarse zoning + template-like priors for remaining (2,3,5)
    std::vector<double> f; zoning(bw, b, 4, f);
    // Heuristic scoring: prefer right-heavy -> 3; top-left to bottom-right curve -> 2; left-heavy bottom -> 5
    double left = 0, right = 0, upper = 0, lower = 0;
    for(int gi=0; gi<4; ++gi){
        for(int gj=0; gj<4; ++gj){
            double v = f[gi*4+gj];
            if (gj < 2) left += v; else right += v;
            if (gi < 2) upper += v; else lower += v;
        }
    }
    if (right > left * 1.15) return 3;
    if (upper > lower * 1.20) return 2; // more mass upper, and ends to right
    // default lean to 5
    return 5;
}

} // namespace nr_internal

int judge(IMAGE_T &img){
    using namespace nr_internal;
    std::vector< std::vector<unsigned char> > bw = binarize(img);
    return classify(bw);
}
