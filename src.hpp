// Minimal classifier: defines only judge with fully-qualified std::vector type
#include <vector>
#include <cmath>

static inline int _imin(int a,int b){return a<b?a:b;}
static inline int _imax(int a,int b){return a>b?a:b;}
static inline double _absd(double x){return x<0?-x:x;}

int judge(std::vector< std::vector<double> > &img){
    int n = (int)img.size();
    int m = n? (int)img[0].size() : 0;
    if (n <= 0 || m <= 0) return 0;

    // Threshold ~ mid of range, floored at 0.35
    double mn = 1e9, mx = -1e9; int cnt=0;
    for(int i=0;i<n;i++) for(int j=0;j<m;j++){
        double v = img[i][j]; if (v<mn) mn=v; if (v>mx) mx=v; cnt++;
    }
    double thr = (mn+mx)*0.5; if (thr < 0.35) thr = 0.35; if (thr > 1.0) thr = 1.0;

    // Binarize to 28x28
    unsigned char bw[28][28]; for(int i=0;i<28;i++) for(int j=0;j<28;j++) bw[i][j]=0;
    for(int i=0;i<n && i<28;i++) for(int j=0;j<m && j<28;j++) bw[i][j] = (img[i][j] >= thr) ? 1 : 0;

    // Any foreground?
    bool anyfg=false; for(int i=0;i<n;i++){ for(int j=0;j<m;j++){ if (bw[i][j]) { anyfg=true; break; } } if (anyfg) break; }
    if (!anyfg) return 1;

    // Bounding box
    int r0=n, c0=m, r1=-1, c1=-1;
    for(int i=0;i<n;i++) for(int j=0;j<m;j++) if (bw[i][j]){ if(i<r0)r0=i; if(j<c0)c0=j; if(i>r1)r1=i; if(j>c1)c1=j; }
    if (r1<r0){ r0=0; c0=0; r1=n-1; c1=m-1; }
    int h=r1-r0+1, w=c1-c0+1; if(h<=0||w<=0) return 1;

    // Mass and ratio
    int mass=0; for(int i=r0;i<=r1;i++) for(int j=c0;j<=c1;j++) if (bw[i][j]) mass++;
    double ratio = w / (double)(h>0?h:1);

    // Projections within bbox
    int prow[28]; int pcol[28]; for(int i=0;i<28;i++){ prow[i]=0; pcol[i]=0; }
    for(int i=0;i<h;i++) for(int j=0;j<w;j++) if (bw[r0+i][c0+j]){ prow[i]++; pcol[j]++; }
    int max_row=0,max_col=0; for(int i=0;i<h;i++) if(prow[i]>max_row) max_row=prow[i]; for(int j=0;j<w;j++) if(pcol[j]>max_col) max_col=pcol[j];

    // Holes (BFS on background inside bbox)
    unsigned char vis[28][28]; for(int i=0;i<28;i++) for(int j=0;j<28;j++) vis[i][j]=0;
    int dr[4]={-1,1,0,0}, dc[4]={0,0,-1,1};
    int qrr[28*28], qcc[28*28]; int qh,qt;
    // mark external background
    for(int i=0;i<h;i++) for(int j=0;j<w;j++) if (i==0||j==0||i==h-1||j==w-1){ if(!bw[r0+i][c0+j] && !vis[i][j]){ qh=qt=0; qrr[qt]=i; qcc[qt]=j; qt++; vis[i][j]=1; while(qh<qt){ int r=qrr[qh], c=qcc[qh]; qh++; for(int k=0;k<4;k++){ int nr=r+dr[k], nc=c+dc[k]; if(nr>=0&&nr<h&&nc>=0&&nc<w && !vis[nr][nc] && !bw[r0+nr][c0+nc]){ vis[nr][nc]=1; qrr[qt]=nr; qcc[qt]=nc; qt++; } } } } }
    int holes=0, holes_area=0; long long sumhr=0,sumhc=0;
    for(int i=0;i<h;i++) for(int j=0;j<w;j++) if(!bw[r0+i][c0+j] && !vis[i][j]){ holes++; qh=qt=0; qrr[qt]=i; qcc[qt]=j; qt++; vis[i][j]=2; int area=0; long long srs=0,scs=0; while(qh<qt){ int r=qrr[qh], c=qcc[qh]; qh++; area++; srs+=r; scs+=c; for(int k=0;k<4;k++){ int nr=r+dr[k], nc=c+dc[k]; if(nr>=0&&nr<h&&nc>=0&&nc<w && !bw[r0+nr][c0+nc] && vis[nr][nc]==0){ vis[nr][nc]=2; qrr[qt]=nr; qcc[qt]=nc; qt++; } } } holes_area += area; sumhr += srs; sumhc += scs; }

    if (holes >= 2) return 8;
    if (holes == 1){
        double hole_y = (h>0)? ((double)sumhr/holes_area) / h : 0.0;
        double hole_x = (w>0)? ((double)sumhc/holes_area) / w : 0.0;
        double hole_frac = (double)holes_area / (double)(h*w>0?h*w:1);
        if (ratio>0.75 && ratio<1.25 && hole_frac>0.10 && _absd(hole_x-0.5)<0.18 && _absd(hole_y-0.5)<0.18) return 0;
        if (hole_y >= 0.5) return 6; else return 9;
    }

    if (ratio < 0.45 && max_col > (_imax(2, h/3))) return 1;
    int top_q = _imin(h, _imax(2, h/4)); int top_sum=0, rest_sum=0; for(int i=0;i<top_q;i++) top_sum+=prow[i]; for(int i=top_q;i<h;i++) rest_sum+=prow[i]; if (top_sum > rest_sum && ratio > 0.7) return 7;
    int midr = h/2; int midr_sum = (midr>=0 && midr<h) ? prow[midr] : 0; int left_col = (w>0) ? pcol[0] : 0; if (midr_sum > mass/5 && left_col > mass/6 && ratio < 1.0) return 4;

    // Coarse zoning 4x4 without normalization
    double left=0,right=0,upper=0,lower=0;
    for(int i=0;i<h;i++) for(int j=0;j<w;j++) if (bw[r0+i][c0+j]){ int gi = (h>0)? (i*4/h) : 0; int gj = (w>0)? (j*4/w) : 0; if (gj<2) left+=1; else right+=1; if (gi<2) upper+=1; else lower+=1; }
    if (right > left * 1.15) return 3;
    if (upper > lower * 1.20) return 2;
    return 5;
}
