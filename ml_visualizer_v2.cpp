#include <iostream>
#include <vector>
#include <queue>
#include <string>
#include <cmath>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <thread>
#include <chrono>
#include <limits>
#include <memory>
#include <sstream>

// ── ANSI ────────────────────────────────────────────────────────────────────
#define R   "\033[0m"
#define B   "\033[1m"
#define DIM "\033[2m"
#define RED     "\033[31m"
#define GRN     "\033[32m"
#define YLW     "\033[33m"
#define BLU     "\033[34m"
#define MAG     "\033[35m"
#define CYN     "\033[36m"
#define WHT     "\033[37m"
#define BGRN    "\033[42m"
#define BRED    "\033[41m"
#define BYEL    "\033[43m"
#define BMAG    "\033[45m"
#define BCYN    "\033[46m"

void sleep_ms(int ms){ std::this_thread::sleep_for(std::chrono::milliseconds(ms)); }
void cls(){ std::cout<<"\033[2J\033[H"; std::cout.flush(); }
void gotoxy(int r,int c){ std::cout<<"\033[" <<r<<";"<<c<<"H"; }
void saveCursor(){ std::cout<<"\033[s"; }
void restCursor(){ std::cout<<"\033[u"; }
void hideCursor(){ std::cout<<"\033[?25l"; }
void showCursor(){ std::cout<<"\033[?25h"; }

// ── Utility ──────────────────────────────────────────────────────────────────
void pressEnter(){
    showCursor();
    std::cout<<"\n"<<CYN<<"  [ Press ENTER to continue... ]"<<R;
    std::cin.ignore(std::numeric_limits<std::streamsize>::max(),'\n');
    std::cin.get();
    hideCursor();
}

void hline(char ch='-', int w=64, const std::string& col=CYN){
    std::cout<<col<<std::string(w,ch)<<R<<"\n";
}

void header(const std::string& title, const std::string& col=CYN){
    hline('=',64,col);
    int pad=(64-(int)title.size())/2;
    std::cout<<col<<B<<std::string(pad,' ')<<title<<R<<"\n";
    hline('=',64,col);
    std::cout<<"\n";
}

// Progress bar helper
std::string pbar(double val, int width=20, const std::string& col=GRN){
    int fill=(int)(val*width);
    fill=std::max(0,std::min(width,fill));
    return col+std::string(fill,'|')+DIM+std::string(width-fill,'.')+R;
}

// ════════════════════════════════════════════════════════════════════════════
// ABSTRACT BASE
// ════════════════════════════════════════════════════════════════════════════
class Algorithm {
public:
    virtual void run()=0;
    virtual std::string name() const=0;
    virtual ~Algorithm()=default;
};

// ════════════════════════════════════════════════════════════════════════════
// 1.  LINEAR REGRESSION  –  animated gradient descent
// ════════════════════════════════════════════════════════════════════════════
class LinearRegression : public Algorithm {
    struct Pt { double x,y; };
    std::vector<Pt> data;
    double w=0.2, b=0.05, lr=0.08;
    static const int GW=56, GH=20;

    void genData(){
        srand(42); data.clear();
        for(int i=0;i<14;i++){
            double x=0.05+(i/14.0)*0.90;
            double noise=((rand()%200)-100)/500.0;
            double y=std::max(0.05,std::min(0.95, 0.25+0.6*x+noise));
            data.push_back({x,y});
        }
    }

    // map [0,1] → grid col/row
    int toCol(double x){ return 1+(int)(x*(GW-2)); }
    int toRow(double y){ return 1+(int)((1-y)*(GH-2)); }

    double loss(){
        double s=0; for(auto&p:data){double e=p.y-(w*p.x+b);s+=e*e;}
        return s/data.size();
    }
    void step(){
        double dw=0,db=0; int n=data.size();
        for(auto&p:data){double e=(w*p.x+b)-p.y; dw+=e*p.x; db+=e;}
        w-=lr*(2.0/n)*dw; b-=lr*(2.0/n)*db;
    }

    void drawFrame(int epoch, double L, const std::vector<double>& hist){
        // Build char grid
        std::vector<std::string> color(GH*GW, DIM);
        std::vector<char>        cell(GH*GW,' ');

        auto idx=[&](int r,int c){return r*GW+c;};
        auto setc=[&](int r,int c,char ch,const std::string&co){
            if(r>=0&&r<GH&&c>=0&&c<GW){cell[idx(r,c)]=ch;color[idx(r,c)]=co;}
        };

        // Border
        for(int c=0;c<GW;c++){setc(0,c,'-',CYN);setc(GH-1,c,'-',CYN);}
        for(int r=0;r<GH;r++){setc(r,0,'|',CYN);setc(r,GW-1,'|',CYN);}
        setc(0,0,'+',CYN);setc(0,GW-1,'+',CYN);
        setc(GH-1,0,'+',CYN);setc(GH-1,GW-1,'+',CYN);

        // Regression line (draw first so points overwrite)
        for(int c=1;c<GW-1;c++){
            double px=(double)c/(GW-1);
            double py=w*px+b;
            int r=toRow(py);
            if(r>0&&r<GH-1) setc(r,c,'~',MAG);
        }

        // Residuals + points
        for(auto&p:data){
            int pc=toCol(p.x), pr=toRow(p.y);
            int lr2=toRow(w*p.x+b);
            double err=std::abs(p.y-(w*p.x+b));
            std::string col=(err<0.04)?GRN:(err<0.10)?YLW:RED;
            // residual
            int r0=std::min(pr,lr2), r1=std::max(pr,lr2);
            for(int r=r0;r<=r1;r++) if(cell[idx(r,pc)]==' ') setc(r,pc,':',DIM);
            // point
            setc(pr,pc,'O',col);
        }

        // Print grid
        for(int r=0;r<GH;r++){
            std::cout<<"  ";
            for(int c=0;c<GW;c++) std::cout<<color[idx(r,c)]<<cell[idx(r,c)]<<R;
            std::cout<<"\n";
        }

        // Stats row
        std::cout<<"\n";
        std::cout<<"  "<<CYN<<"Epoch "<<B<<std::setw(3)<<epoch<<R;
        std::cout<<CYN<<"  w="<<B<<MAG<<std::fixed<<std::setprecision(4)<<w<<R;
        std::cout<<CYN<<"  b="<<B<<MAG<<std::setprecision(4)<<b<<R;
        std::cout<<CYN<<"  Loss="<<B<<RED<<std::setprecision(5)<<L<<R<<"\n";

        // Legend
        std::cout<<"  "<<GRN<<"O"<<R<<" low err  "<<YLW<<"O"<<R<<" med  "<<RED<<"O"<<R<<" high  "
                 <<MAG<<"~"<<R<<" regression line\n";

        // Loss sparkline
        std::cout<<"\n  Loss trend: ";
        int show=std::min((int)hist.size(),30);
        double mx=hist.empty()?1:hist[0];
        for(int i=(int)hist.size()-show;i<(int)hist.size();i++){
            double norm=hist[i]/mx;
            const char* bars[]={" ","▁","▂","▃","▄","▅","▆","▇","█"};
            int bi=(int)((1-norm)*8);
            std::string col=norm<0.3?GRN:norm<0.6?YLW:RED;
            std::cout<<col<<bars[bi]<<R;
        }
        std::cout<<"  (falling = learning)\n";
    }

public:
    std::string name() const override { return "Linear Regression (Gradient Descent)"; }
    void run() override {
        genData(); w=0.2; b=0.05;
        hideCursor();
        cls(); header("LINEAR REGRESSION — Gradient Descent",MAG);
        std::cout<<CYN<<"  14 noisy points  |  true: y = 0.6x + 0.25  |  lr="<<lr<<R<<"\n";
        std::cout<<"  Watch the line fit and residuals shrink!\n";
        pressEnter();

        std::vector<double> hist;
        for(int ep=1;ep<=50;ep++){
            cls(); header("LINEAR REGRESSION — Gradient Descent",MAG);
            double L=loss(); hist.push_back(L);
            drawFrame(ep,L,hist);
            step();
            sleep_ms(130);
        }
        std::cout<<"\n  "<<GRN<<B<<"✓ Done! Line converged."<<R<<"\n";
        pressEnter();
    }
};

// ════════════════════════════════════════════════════════════════════════════
// 2.  K-MEANS  –  with Voronoi regions drawn per cell
// ════════════════════════════════════════════════════════════════════════════
class KMeans : public Algorithm {
    struct Pt{ double x,y; int cls; };
    static const int K=3, GW=56, GH=20;
    std::vector<Pt> pts;
    double cx[K],cy[K];
    const char SYM[K]={'*','#','+'};
    const std::string COL[K]={RED,GRN,BLU};

    void gen(){
        srand(77); pts.clear();
        double bx[]={0.22,0.78,0.50}, by[]={0.75,0.75,0.22};
        for(int k=0;k<K;k++)
            for(int i=0;i<10;i++){
                double x=bx[k]+((rand()%100)-50)/380.0;
                double y=by[k]+((rand()%100)-50)/380.0;
                x=std::max(0.05,std::min(0.95,x));
                y=std::max(0.05,std::min(0.95,y));
                pts.push_back({x,y,-1});
            }
        // fixed initial centroids
        cx[0]=0.15;cy[0]=0.50;
        cx[1]=0.85;cy[1]=0.50;
        cx[2]=0.50;cy[2]=0.90;
    }

    double dist2(double ax,double ay,double bx,double by){
        return (ax-bx)*(ax-bx)+(ay-by)*(ay-by);
    }

    bool assign(){
        bool changed=false;
        for(auto&p:pts){
            int best=0;
            double bd=dist2(p.x,p.y,cx[0],cy[0]);
            for(int k=1;k<K;k++){
                double d=dist2(p.x,p.y,cx[k],cy[k]);
                if(d<bd){bd=d;best=k;}
            }
            if(p.cls!=best){p.cls=best;changed=true;}
        }
        return changed;
    }

    void updateC(){
        for(int k=0;k<K;k++){
            double sx=0,sy=0; int n=0;
            for(auto&p:pts) if(p.cls==k){sx+=p.x;sy+=p.y;n++;}
            if(n){cx[k]=sx/n;cy[k]=sy/n;}
        }
    }

    void draw(int iter, const std::string& phase){
        int GW2=GW, GH2=GH;
        std::vector<char> cell(GH2*GW2,' ');
        std::vector<std::string> col(GH2*GW2,R);
        auto idx=[&](int r,int c){return r*GW2+c;};
        auto set=[&](int r,int c,char ch,const std::string&co){
            if(r>=0&&r<GH2&&c>=0&&c<GW2){cell[idx(r,c)]=ch;col[idx(r,c)]=co;}
        };

        // ── Voronoi background ──
        for(int r=0;r<GH2;r++)
        for(int c=0;c<GW2;c++){
            double px=(double)c/GW2, py=1.0-(double)r/GH2;
            int best=0; double bd=dist2(px,py,cx[0],cy[0]);
            for(int k=1;k<K;k++){double d=dist2(px,py,cx[k],cy[k]);if(d<bd){bd=d;best=k;}}
            set(r,c,'.',DIM+COL[best]);
        }

        // Border
        for(int c=0;c<GW2;c++){set(0,c,'-',CYN);set(GH2-1,c,'-',CYN);}
        for(int r=0;r<GH2;r++){set(r,0,'|',CYN);set(r,GW2-1,'|',CYN);}
        set(0,0,'+',CYN);set(0,GW2-1,'+',CYN);
        set(GH2-1,0,'+',CYN);set(GH2-1,GW2-1,'+',CYN);

        // Data points
        for(auto&p:pts){
            int pc=1+(int)(p.x*(GW2-2));
            int pr=1+(int)((1-p.y)*(GH2-2));
            if(p.cls>=0) set(pr,pc,SYM[p.cls],B+COL[p.cls]);
            else set(pr,pc,'?',DIM);
        }

        // Centroids (big @)
        for(int k=0;k<K;k++){
            int pc=1+(int)(cx[k]*(GW2-2));
            int pr=1+(int)((1-cy[k])*(GH2-2));
            set(pr,pc,'@',B+COL[k]);
        }

        // Print
        for(int r=0;r<GH2;r++){
            std::cout<<"  ";
            for(int c=0;c<GW2;c++) std::cout<<col[idx(r,c)]<<cell[idx(r,c)]<<R;
            std::cout<<"\n";
        }

        std::cout<<"\n  "<<CYN<<"Iter "<<B<<iter<<R<<"  │  "<<YLW<<B<<phase<<R<<"\n\n";
        for(int k=0;k<K;k++){
            int n=0; for(auto&p:pts) if(p.cls==k) n++;
            std::cout<<"  "<<COL[k]<<B<<"@ Centroid "<<k+1<<R
                     <<"  x="<<std::fixed<<std::setprecision(3)<<cx[k]
                     <<" y="<<cy[k]
                     <<"  members="<<B<<n<<R<<"\n";
        }
    }

public:
    std::string name() const override { return "K-Means Clustering (K=3)"; }
    void run() override {
        gen();
        hideCursor();
        cls(); header("K-MEANS CLUSTERING  (K=3)",BLU);
        std::cout<<CYN<<"  30 points in 3 natural clusters\n";
        std::cout<<"  Voronoi regions shown as background\n";
        std::cout<<"  @ = centroid   * # + = data points\n"<<R;
        pressEnter();

        bool changed=true;
        for(int it=1;it<=12&&changed;it++){
            // Assign
            cls(); header("K-MEANS CLUSTERING  (K=3)",BLU);
            changed=assign();
            draw(it,"Assigning points → nearest centroid");
            sleep_ms(700);
            // Update
            cls(); header("K-MEANS CLUSTERING  (K=3)",BLU);
            updateC();
            draw(it,"Moving centroids → cluster mean");
            sleep_ms(700);
        }
        std::cout<<"\n  "<<GRN<<B<<"✓ Converged!"<<R<<"\n";
        pressEnter();
    }
};

// ════════════════════════════════════════════════════════════════════════════
// 3.  DECISION TREE  –  proper 2-D ASCII tree with branches
// ════════════════════════════════════════════════════════════════════════════
struct Node {
    bool leaf=false;
    std::string feat, label;
    double thr=0, gini=0;
    int samples=0;
    std::unique_ptr<Node> left,right;
};

class DecisionTree : public Algorithm {
    std::unique_ptr<Node> root;

    void build(){
        auto n=[](bool lf,std::string f,std::string lb,double t,double g,int s)->std::unique_ptr<Node>{
            auto nd=std::make_unique<Node>();
            nd->leaf=lf;nd->feat=f;nd->label=lb;nd->thr=t;nd->gini=g;nd->samples=s;
            return nd;
        };
        root=n(false,"petal_len","",2.45,0.667,150);
        root->left =n(true,"","Setosa",0,0.000,50);
        root->right=n(false,"petal_wid","",1.75,0.500,100);
        root->right->left =n(false,"petal_len","",4.95,0.168,54);
        root->right->left->left =n(true,"","Versicolor",0,0.000,47);
        root->right->left->right=n(true,"","Virginica",0,0.444,7);
        root->right->right=n(false,"petal_len","",4.85,0.043,46);
        root->right->right->left =n(true,"","Versicolor",0,0.444,3);
        root->right->right->right=n(true,"","Virginica",0,0.000,43);
    }

    std::string nodeColor(const Node*n){
        if(!n)return R;
        if(n->leaf){
            if(n->label=="Setosa")return GRN;
            if(n->label=="Versicolor")return YLW;
            return MAG;
        }
        return CYN;
    }

    // ── 2-D grid-based tree drawing ──────────────────────────────────────
    struct Cell{ std::string text; std::string col; };
    static const int COLS=72, ROWS=22;
    std::vector<Cell> canvas;

    void canvasClear(){
        canvas.assign(COLS*ROWS,{"",R});
    }
    void put(int r,int c,const std::string&s,const std::string&col=""){
        if(r<0||r>=ROWS||c<0||c+(int)s.size()>COLS) return;
        for(int i=0;i<(int)s.size();i++){
            int idx=r*COLS+c+i;
            if(idx>=0&&idx<(int)canvas.size()){
                canvas[idx].text=std::string(1,s[i]);
                canvas[idx].col=col.empty()?R:col;
            }
        }
    }
    void putc2(int r,int c,char ch,const std::string&col=R){
        if(r>=0&&r<ROWS&&c>=0&&c<COLS){
            canvas[r*COLS+c].text=std::string(1,ch);
            canvas[r*COLS+c].col=col;
        }
    }
    void vline(int r0,int r1,int c,const std::string&col){
        for(int r=r0;r<=r1;r++) putc2(r,c,'|',col);
    }
    void hlineC(int r,int c0,int c1,const std::string&col){
        for(int c=c0;c<=c1;c++) putc2(r,c,'-',col);
    }
    void printCanvas(){
        for(int r=0;r<ROWS;r++){
            std::cout<<"  ";
            for(int c=0;c<COLS;c++){
                auto&cell=canvas[r*COLS+c];
                if(cell.text.empty()) std::cout<<' ';
                else std::cout<<cell.col<<cell.text<<R;
            }
            std::cout<<"\n";
        }
    }

    // Place node box; return center column
    int placeNode(int row, int col, const Node*nd, const std::string& overrideCol=""){
        std::string co=overrideCol.empty()?nodeColor(nd):overrideCol;
        std::string txt;
        if(nd->leaf) txt="["+nd->label+"]";
        else         txt=nd->feat+"<="+std::to_string(nd->thr).substr(0,4);
        int w=(int)txt.size()+2;
        int c0=col-w/2;
        // box
        put(row,c0,"+"+std::string(w,'-')+"+",co);
        put(row+1,c0,"|"+txt+"|",co);
        put(row+2,c0,"+"+std::string(w,'-')+"+",co);
        // gini+samples below
        std::string info="g="+std::to_string(nd->gini).substr(0,4)+" n="+std::to_string(nd->samples);
        put(row+3,col-(int)info.size()/2,info,DIM);
        return col;
    }

    // Recursive layout: returns center column used
    int layout(const Node*nd, int row, int lc, int rc){
        if(!nd) return (lc+rc)/2;
        int mid=(lc+rc)/2;
        if(nd->leaf){
            placeNode(row,mid,nd);
            return mid;
        }
        // children on row+5
        int lMid=layout(nd->left.get(), row+5, lc, mid-1);
        int rMid=layout(nd->right.get(),row+5, mid+1, rc);
        // place this node above mid of children
        int nmid=(lMid+rMid)/2;
        placeNode(row,nmid,nd);
        // draw connectors
        std::string co=nodeColor(nd);
        vline(row+3,row+4,nmid,co);
        hlineC(row+4,lMid,rMid,co);
        putc2(row+4,lMid,'/',GRN);
        putc2(row+4,rMid,'\\',YLW);
        vline(row+3,row+4,lMid,GRN);
        vline(row+3,row+4,rMid,YLW);
        // Yes/No labels
        put(row+4,lMid-3,"Yes",GRN);
        put(row+4,rMid+1,"No ",YLW);
        return nmid;
    }

    // ── inference path ──────────────────────────────────────────────────
    void predict(double pl, double pw){
        std::cout<<"\n  "<<CYN<<B<<"Sample: petal_len="<<pl<<"  petal_wid="<<pw<<R<<"\n\n";
        Node*cur=root.get(); int step=1;
        while(cur){
            if(cur->leaf){
                std::cout<<"  "<<GRN<<B<<"→ CLASS: "<<cur->label<<R<<"\n"; break;
            }
            double val=(cur->feat=="petal_len")?pl:pw;
            bool go=val<=cur->thr;
            std::string dir=go?std::string(GRN)+"YES (left)":std::string(YLW)+"NO  (right)";
            std::cout<<"  "<<step++<<". "<<CYN<<cur->feat<<R<<" = "<<val
                     <<(go?" <= ":" >  ")<<cur->thr<<"  →  "<<dir<<R<<"\n";
            sleep_ms(350);
            cur=go?cur->left.get():cur->right.get();
        }
    }

    // ── BFS ─────────────────────────────────────────────────────────────
    void bfs(){
        std::queue<const Node*> q; q.push(root.get()); int lv=0;
        while(!q.empty()){
            int sz=q.size();
            std::cout<<"  "<<YLW<<B<<"L"<<lv<<": "<<R;
            for(int i=0;i<sz;i++){
                auto*n=q.front();q.pop();
                std::string co=nodeColor(n);
                if(n->leaf) std::cout<<co<<"["<<n->label<<"] "<<R;
                else        std::cout<<co<<"("<<n->feat<<"<="<<n->thr<<") "<<R;
                if(n->left)  q.push(n->left.get());
                if(n->right) q.push(n->right.get());
            }
            std::cout<<"\n"; sleep_ms(400); lv++;
        }
    }

public:
    std::string name() const override { return "Decision Tree (Iris)"; }
    void run() override {
        build();
        hideCursor();

        // ── Step 1: 2-D tree drawing ──
        cls(); header("DECISION TREE — Iris Dataset",GRN);
        std::cout<<"  "<<CYN<<"Step 1: Full tree (2-D layout with branches)\n\n"<<R;
        canvasClear();
        layout(root.get(),0,0,COLS-1);
        printCanvas();
        pressEnter();

        // ── Step 2: BFS ──
        cls(); header("DECISION TREE — Iris Dataset",GRN);
        std::cout<<"  "<<CYN<<"Step 2: BFS level-order traversal\n\n"<<R;
        bfs();
        pressEnter();

        // ── Step 3: Inference ──
        cls(); header("DECISION TREE — Iris Dataset",GRN);
        std::cout<<"  "<<CYN<<"Step 3: Inference — trace the decision path\n"<<R;
        predict(1.4,0.2);
        predict(4.8,1.6);
        predict(6.1,2.3);
        pressEnter();
    }
};

// ════════════════════════════════════════════════════════════════════════════
// 4.  NEURAL NETWORK  –  proper layered diagram with connections
// ════════════════════════════════════════════════════════════════════════════
class NeuralNet : public Algorithm {
    std::vector<int> arch={3,4,4,2};
    std::vector<std::vector<double>> act;
    std::vector<std::vector<std::vector<double>>> W;

    double sigmoid(double x){return 1.0/(1+std::exp(-x));}
    double relu(double x){return x>0?x:0;}

    void initW(){
        srand(13); W.clear();
        for(int l=0;l<(int)arch.size()-1;l++){
            std::vector<std::vector<double>> lw;
            for(int j=0;j<arch[l+1];j++){
                std::vector<double> row;
                for(int i=0;i<arch[l];i++)
                    row.push_back(((rand()%200)-100)/150.0);
                lw.push_back(row);
            }
            W.push_back(lw);
        }
    }

    void forward(std::vector<double> inp){
        act.clear(); act.push_back(inp);
        for(int l=0;l<(int)W.size();l++){
            std::vector<double> nxt;
            for(int j=0;j<(int)W[l].size();j++){
                double z=0;
                for(int i=0;i<(int)act[l].size();i++) z+=W[l][j][i]*act[l][i];
                nxt.push_back(l==(int)W.size()-1?sigmoid(z):relu(z));
            }
            act.push_back(nxt);
        }
    }

    // ── Full network diagram on one screen ──────────────────────────────
    // We'll use a char canvas, draw nodes + connections
    static const int CW=70, CH=32;
    struct CC{ char ch=' '; std::string col=R; };
    std::vector<CC> cnv;

    void clearCnv(){ cnv.assign(CW*CH,{' ',R}); }
    void setC(int r,int c,char ch,const std::string&co=R){
        if(r>=0&&r<CH&&c>=0&&c<CW){cnv[r*CW+c]={ch,co};}
    }
    void printCnv(){
        for(int r=0;r<CH;r++){
            std::cout<<"  ";
            for(int c=0;c<CW;c++){
                auto&cc=cnv[r*CW+c];
                std::cout<<cc.col<<cc.ch<<R;
            }
            std::cout<<"\n";
        }
    }

    // Draw a node circle/box at (row,col), return label col used
    void drawNode(int r,int c,double activation,bool active,bool done){
        char ch;
        std::string co;
        if(!done&&!active){ ch='o'; co=DIM; }
        else {
            if(activation>0.65){ch='O';co=std::string(B)+GRN;}
            else if(activation>0.35){ch='o';co=YLW;}
            else{ch='.';co=RED;}
        }
        if(active) co=std::string(B)+CYN;
        setC(r,c,'(',co); setC(r,c+1,ch,co); setC(r,c+2,')',co);
    }

    // Draw connection line from (r1,c1) to (r2,c2) with weight
    void drawConn(int r1,int c1,int r2,int c2,double weight,bool highlight){
        std::string co;
        if(!highlight){co=DIM;}
        else{ co=weight>0?GRN:RED; }

        // Simple straight-ish line with intermediate points
        int dc=c2-c1, dr=r2-r1;
        int steps=std::max(std::abs(dc),std::abs(dr));
        if(steps==0)return;
        for(int s=1;s<steps;s++){
            int r=r1+(int)std::round((double)s*dr/steps);
            int c=c1+(int)std::round((double)s*dc/steps);
            char ch=std::abs(dr)>std::abs(dc)*0.6?'|':'-';
            if(cnv[r*CW+c].ch==' ') setC(r,c,ch,co);
        }
    }

    // Compute positions for each layer
    std::vector<std::vector<std::pair<int,int>>> nodePos(){
        std::vector<std::vector<std::pair<int,int>>> pos;
        int numL=arch.size();
        // columns evenly spaced, leaving 3 chars per node
        for(int l=0;l<numL;l++){
            int col=5+l*(CW-10)/(numL-1);
            int n=arch[l];
            std::vector<std::pair<int,int>> lpos;
            for(int j=0;j<n;j++){
                int row=3+j*(CH-6)/(n>1?n-1:1);
                lpos.push_back({row,col});
            }
            pos.push_back(lpos);
        }
        return pos;
    }

    void drawNetwork(int highlightLayer){
        clearCnv();
        auto pos=nodePos();
        int numL=arch.size();

        // Draw all connections first (background)
        for(int l=0;l<numL-1;l++){
            bool hl=(l==highlightLayer-1);
            for(auto&[r1,c1]:pos[l])
            for(auto&[r2,c2]:pos[l+1]){
                double wval=0;
                if(hl&&!act.empty()) wval=W[l][&r2-&pos[l+1][0].first][&r1-&pos[l][0].first];
                // just use highlight flag; weight indexing simplified
                drawConn(r1,c1+1,r2,c2+1,1.0,hl);
            }
        }

        // Draw weights on active layer connections with values
        if(highlightLayer>0&&!act.empty()){
            int l=highlightLayer-1;
            for(int j=0;j<(int)pos[l+1].size();j++){
                for(int i=0;i<(int)pos[l].size();i++){
                    auto[r1,c1]=pos[l][i];
                    auto[r2,c2]=pos[l+1][j];
                    double wv=W[l][j][i];
                    std::string co=wv>0?GRN:RED;
                    // place weight value at midpoint
                    int mr=(r1+r2)/2, mc=(c1+c2)/2;
                    // format short
                    std::ostringstream ss; ss<<std::fixed<<std::setprecision(1)<<wv;
                    std::string ws=ss.str();
                    for(int k=0;k<(int)ws.size();k++) if(mr>=0&&mr<CH&&mc+k>=0&&mc+k<CW)
                        setC(mr,mc+k,ws[k],co);
                }
            }
        }

        // Draw nodes on top
        for(int l=0;l<numL;l++){
            bool active=(l==highlightLayer);
            bool done=(l<highlightLayer);
            for(int j=0;j<(int)pos[l].size();j++){
                auto[r,c]=pos[l][j];
                double av=act.empty()?0.5:act[l][j];
                drawNode(r,c,av,active,done);
                // activation value below node
                if(!act.empty()){
                    std::ostringstream ss; ss<<std::fixed<<std::setprecision(2)<<av;
                    std::string vs=ss.str();
                    std::string co=av>0.65?GRN:av>0.35?YLW:RED;
                    if(done||active) setC(r+1,c,vs[0],co),setC(r+1,c+1,vs[1],co),
                                     setC(r+1,c+2,vs[2],co),setC(r+1,c+3,vs[3],co);
                }
            }
        }

        // Layer labels
        std::vector<std::string> lnames={"Input","Hid-1","Hid-2","Output"};
        for(int l=0;l<numL;l++){
            int c=pos[l][0].second;
            std::string co=(l==highlightLayer)?std::string(B)+CYN:DIM;
            int r=CH-2;
            for(int k=0;k<(int)lnames[l].size();k++) setC(r,c+k-1,lnames[l][k],co);
        }

        printCnv();
    }

public:
    std::string name() const override { return "Neural Network (Forward Pass)"; }
    void run() override {
        initW();
        hideCursor();
        cls(); header("NEURAL NETWORK — Forward Pass [3→4→4→2]",MAG);
        std::cout<<CYN<<"  Architecture : Input(3) → Hidden(4,ReLU) → Hidden(4,ReLU) → Output(2,Sigmoid)\n";
        std::cout<<"  Connections  : shown with weights (green=+, red=-)\n";
        std::cout<<"  Nodes        : (O)=high  (o)=mid  (.)=low activation\n"<<R;
        pressEnter();

        std::vector<double> inp={0.9,0.4,0.7};
        forward(inp);

        std::vector<std::string> lnames={"Input layer","Hidden layer 1","Hidden layer 2","Output layer"};
        std::vector<std::string> descs={
            "Raw input features fed in",
            "z = W1*x + b1  then ReLU(z)",
            "z = W2*h1 + b2  then ReLU(z)",
            "z = W3*h2 + b3  then Sigmoid(z)"
        };

        for(int l=0;l<(int)arch.size();l++){
            cls(); header("NEURAL NETWORK — Forward Pass [3→4→4→2]",MAG);
            drawNetwork(l);
            std::cout<<"\n  "<<B<<CYN<<"► "<<lnames[l]<<R<<"  —  "<<descs[l]<<"\n";
            // print activations
            std::cout<<"  Activations: ";
            for(double a:act[l]){
                std::string co=a>0.65?GRN:a>0.35?YLW:RED;
                std::cout<<co<<std::fixed<<std::setprecision(3)<<a<<R<<"  ";
            }
            std::cout<<"\n";
            // print weight matrix for active transition
            if(l>0){
                std::cout<<"\n  Weight matrix W"<<l<<" ("<<W[l-1].size()<<"x"<<W[l-1][0].size()<<"):\n";
                for(auto&row:W[l-1]){
                    std::cout<<"  │";
                    for(double wv:row){
                        std::string co=wv>0.3?GRN:wv<-0.3?RED:YLW;
                        std::cout<<co<<std::setw(7)<<std::fixed<<std::setprecision(3)<<wv<<R;
                    }
                    std::cout<<"  │\n";
                }
            }
            sleep_ms(500);
            pressEnter();
        }

        // Final output
        cls(); header("NEURAL NETWORK — Forward Pass [3→4→4→2]",MAG);
        drawNetwork((int)arch.size()-1);
        std::cout<<"\n  "<<B<<GRN<<"FINAL OUTPUT:\n"<<R;
        auto&out=act.back();
        std::vector<std::string> cls2={"Class A","Class B"};
        int pred=out[0]>out[1]?0:1;
        for(int i=0;i<(int)out.size();i++){
            std::string co=(i==pred)?std::string(B)+GRN:RED;
            int bar=(int)(out[i]*36);
            std::cout<<"  "<<cls2[i]<<" │"<<co<<std::string(bar,'|')<<R
                     <<std::string(36-bar,'.')
                     <<"│ "<<std::fixed<<std::setprecision(4)<<out[i]
                     <<(i==pred?" ◄ predicted":"")<<"\n";
        }
        std::cout<<"\n  Confidence: "<<B<<(int)(std::max(out[0],out[1])*100)<<"%"<<R<<"\n";
        pressEnter();
    }
};

// ════════════════════════════════════════════════════════════════════════════
// MENU
// ════════════════════════════════════════════════════════════════════════════
class Menu {
    std::vector<std::unique_ptr<Algorithm>> algos;
public:
    Menu(){
        algos.push_back(std::make_unique<LinearRegression>());
        algos.push_back(std::make_unique<KMeans>());
        algos.push_back(std::make_unique<DecisionTree>());
        algos.push_back(std::make_unique<NeuralNet>());
    }
    void show(){
        while(true){
            showCursor(); cls();
            std::cout<<CYN<<B;
            std::cout<<"╔══════════════════════════════════════════════════╗\n";
            std::cout<<"║         ML ALGORITHM VISUALIZER  v2  (C++)      ║\n";
            std::cout<<"║           OOP + Data Structures Edition          ║\n";
            std::cout<<"╚══════════════════════════════════════════════════╝\n"<<R;
            std::cout<<"\n  "<<YLW<<"Pick an algorithm:\n\n"<<R;
            for(int i=0;i<(int)algos.size();i++)
                std::cout<<"  "<<YLW<<B<<"["<<i+1<<"]"<<R<<"  "<<algos[i]->name()<<"\n";
            std::cout<<"\n  "<<RED<<B<<"[0]"<<R<<"  Exit\n\n"<<CYN<<"  > "<<R;
            int ch; std::cin>>ch;
            if(ch==0){ std::cout<<GRN<<"\n  Goodbye!\n"<<R; showCursor(); break; }
            if(ch>=1&&ch<=(int)algos.size()) algos[ch-1]->run();
        }
    }
};

int main(){ Menu m; m.show(); return 0; }
