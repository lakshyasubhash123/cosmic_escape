#ifdef __APPLE__
  #include <OpenGL/gl.h>
  #include <OpenGL/glu.h>
  #include <GLUT/glut.h>
#else
  #include <GL/gl.h>
  #include <GL/glu.h>
  #include <GL/glut.h>
#endif

#include <cmath>
#include <cstdlib>
#include <cstdio>
#include <cstring>

static const float PI     = 3.14159265f;
static const float TWO_PI = 6.28318530f;
static const int   W      = 1280;
static const int   H      = 720;
static const float STEP   = 0.016f;          /* fixed simulation step (s) */

inline float randf(float lo, float hi){
    return lo + (hi-lo)*(float)rand()/(float)RAND_MAX;
}
inline float clampf(float v,float lo,float hi){ return v<lo?lo:v>hi?hi:v; }
inline float lerpf(float a,float b,float t){ return a+(b-a)*t; }
inline float smooth01(float t){ t=clampf(t,0.f,1.f); return t*t*(3.f-2.f*t); }
/* wrap v into [lo,hi) */
static float wrapf(float v,float lo,float hi){
    float r=hi-lo; v=fmodf(v-lo,r); if(v<0.f)v+=r; return v+lo;
}
/* deterministic hash -> [0,1); replaces srand()/rand() calls inside draw code */
static float hash01(int n){
    float s=sinf((float)n*12.9898f)*43758.5453f; return s-floorf(s);
}

static float gTime=0.f;
static int   gPhase=0;
static float gPhaseTime=0.f;
static int   gFrame=0;
static const float PHASE_DUR[]={14.f,10.f,8.f};

/* runtime controls */
static bool  gPaused=false, gShowHUD=true, gLowQ=false, gFull=false, gShotReq=false;
static float gTimeScale=1.f, gAcc=0.f, gFps=60.f, gFade=0.f;
static int   gLastMs=0, gShots=0;
static int   gVpX=0,gVpY=0,gVpW=W,gVpH=H;

/* ── Stars ── */
struct Star{ float x,y,layer,bright,tph,size,r,g,b; };
static const int NS=800;
static Star gS[NS];

/* ── Planets ── */
struct Planet{
    float cx,cy,radius,rotA,rotSpd;
    bool  ring;
    float pr,pg,pb,ar,ag,ab,atmosSc,blobSeed;
};
static const int NP=5;
static Planet gP[NP];

/* ── Asteroids (live in a box that follows the camera, so no pop-in) ── */
struct Asteroid{
    float x,y,vx,vy,bvx,bvy,radius,rotA,rotSpd;
    int nv; float vx2[14],vy2[14];
};
static const int   NA=40;
static const float ARX=1000.f, ARY=560.f;   /* half-size of the asteroid box */
static Asteroid gA[NA];

/* ── Particles ── */
struct Particle{ float x,y,vx,vy,life,maxLife,size,r,g,b,a; int type; };
static const int MP=1400;
static Particle gParts[MP];
static int gNP=0;

/* ── Nebula ── */
struct Nebula{ float x,y,rx,ry,r,g,b,a,phase; };
static const int NN=7;
static Nebula gNeb[NN];

/* ── Ship ── */
struct Ship{
    float x,y,angle,engineGlow,pathT,shrink;
    float cpx[4],cpy[4];
};
static Ship gShip;

/* ── Black hole ── */
struct BH{ float x,y,radius,spin,alpha; };
static BH gBH={W*0.62f,H*0.48f,60.f,0.f,0.f};

/* ── Warp gate ── */
struct WG{ float x,y,radius,spin,alpha,charge; };
static WG gWG={W*0.87f,H*0.50f,80.f,0.f,0.f,0.f};

/* ── Camera ── */
static float gCamOx=0.f,gCamOy=0.f,gCamZoom=1.f,gCamShake=0.f,gShkX=0.f,gShkY=0.f;

/* ═══ DRAWING PRIMITIVES ═══
   cosf/sinf are looked up from cached unit-circle tables (disc/ring/ellipse
   are called thousands of times per frame). */
static const int MAXSEG=128;
static float gCs[MAXSEG+1][MAXSEG+1], gSn[MAXSEG+1][MAXSEG+1];
static bool  gTabOk[MAXSEG+1];
static void circleTab(int& seg,const float*& c,const float*& s){
    if(seg>MAXSEG)seg=MAXSEG;
    if(seg<3)seg=3;
    if(!gTabOk[seg]){
        for(int i=0;i<=seg;i++){
            float a=i*TWO_PI/seg; gCs[seg][i]=cosf(a); gSn[seg][i]=sinf(a);
        }
        gTabOk[seg]=true;
    }
    c=gCs[seg]; s=gSn[seg];
}
static void ellipse(float cx,float cy,float rx,float ry,int seg=52){
    const float *c,*s; circleTab(seg,c,s);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(cx,cy);
    for(int i=0;i<=seg;i++) glVertex2f(cx+c[i]*rx,cy+s[i]*ry);
    glEnd();
}
static void disc(float cx,float cy,float r,int seg=56){ ellipse(cx,cy,r,r,seg); }
static void ring(float cx,float cy,float r,int seg=72){
    const float *c,*s; circleTab(seg,c,s);
    glBegin(GL_LINE_LOOP);
    for(int i=0;i<seg;i++) glVertex2f(cx+c[i]*r,cy+s[i]*r);
    glEnd();
}
/* leaves additive blending ON (callers that were additive stay additive) */
static void glow(float cx,float cy,float r,float cr,float cg,float cb,float ca,int layers=6){
    if(gLowQ) layers=(layers+1)/2;
    glBlendFunc(GL_SRC_ALPHA,GL_ONE);
    for(int i=0;i<layers;i++){
        float t=(float)i/layers;
        glColor4f(cr,cg,cb,ca*(1.f-t)*0.45f);
        disc(cx,cy,r*(1.f+t*2.2f));
    }
}

static void spawnPart(float x,float y,float vx,float vy,
                      float life,float sz,float r,float g,float b,float a,int type){
    if(gNP>=MP)return;
    Particle& p=gParts[gNP++];
    p.x=x;p.y=y;p.vx=vx;p.vy=vy;
    p.life=p.maxLife=life;p.size=sz;
    p.r=r;p.g=g;p.b=b;p.a=a;p.type=type;
}

static void text(float x,float y,void* font,const char* s){
    glRasterPos2f(x,y);
    for(;*s;s++) glutBitmapCharacter(font,*s);
}

/* ═══ INIT ═══ */
static void initStars(){
    static const float pr[]={1.f,1.f,0.7f,0.6f,0.9f};
    static const float pg[]={1.f,0.9f,0.8f,0.8f,0.7f};
    static const float pb[]={1.f,0.7f,0.5f,1.f,1.f};
    for(int i=0;i<NS;i++){
        gS[i].x=randf(0.f,W); gS[i].y=randf(0.f,H);
        gS[i].layer=randf(0.f,1.f); gS[i].bright=randf(0.3f,1.f);
        gS[i].tph=randf(0.f,TWO_PI); gS[i].size=randf(0.4f,2.5f)*(0.5f+gS[i].layer*0.5f);
        int ci=rand()%5; gS[i].r=pr[ci]; gS[i].g=pg[ci]; gS[i].b=pb[ci];
    }
}
static void initPlanets(){
    struct PD{float cx,cy,r;bool ring;float pr,pg,pb,ar,ag,ab;};
    static const PD d[NP]={
        {220,175,55,true, 0.25f,0.50f,1.0f,0.1f,0.3f,0.9f},
        {980,390,68,true, 0.65f,0.20f,0.85f,0.4f,0.05f,0.7f},
        {640,115,36,false,0.15f,0.85f,0.65f,0.05f,0.6f,0.45f},
        {870,565,45,false,0.95f,0.45f,0.15f,0.8f,0.25f,0.05f},
        {140,540,28,false,0.75f,0.80f,1.0f,0.5f,0.6f,0.9f},
    };
    for(int i=0;i<NP;i++){
        gP[i].cx=d[i].cx; gP[i].cy=d[i].cy; gP[i].radius=d[i].r;
        gP[i].rotA=randf(0.f,TWO_PI);
        gP[i].rotSpd=randf(0.002f,0.009f)*(rand()%2?1.f:-1.f);
        gP[i].ring=d[i].ring;
        gP[i].pr=d[i].pr; gP[i].pg=d[i].pg; gP[i].pb=d[i].pb;
        gP[i].ar=d[i].ar; gP[i].ag=d[i].ag; gP[i].ab=d[i].ab;
        gP[i].atmosSc=1.38f+randf(0.f,0.18f);
        gP[i].blobSeed=randf(0.f,100.f);
    }
}
static void initAsteroids(){
    for(int i=0;i<NA;i++){
        gA[i].x=randf(W*0.5f-ARX,W*0.5f+ARX); gA[i].y=randf(H*0.5f-ARY,H*0.5f+ARY);
        gA[i].bvx=gA[i].vx=randf(-0.35f,0.35f); gA[i].bvy=gA[i].vy=randf(-0.30f,0.30f);
        gA[i].radius=randf(10.f,26.f);
        gA[i].rotA=randf(0.f,TWO_PI);
        gA[i].rotSpd=randf(0.008f,0.04f)*(rand()%2?1.f:-1.f);
        gA[i].nv=8+rand()%4;
        for(int j=0;j<gA[i].nv;j++){
            float ang=j*TWO_PI/gA[i].nv+randf(-0.15f,0.15f);
            float r2=0.60f+randf(0.f,0.40f);
            gA[i].vx2[j]=cosf(ang)*r2; gA[i].vy2[j]=sinf(ang)*r2;
        }
    }
}
static void initNebula(){
    static const float nr[]={0.3f,0.1f,0.55f,0.05f,0.7f,0.15f,0.0f};
    static const float ng[]={0.1f,0.2f,0.05f,0.45f,0.1f,0.60f,0.5f};
    static const float nb[]={0.7f,0.85f,0.8f,0.80f,0.3f,0.90f,0.8f};
    for(int i=0;i<NN;i++){
        gNeb[i].x=randf(0.f,W); gNeb[i].y=randf(0.f,H);
        gNeb[i].rx=randf(160.f,380.f); gNeb[i].ry=randf(90.f,240.f);
        gNeb[i].r=nr[i]; gNeb[i].g=ng[i]; gNeb[i].b=nb[i];
        gNeb[i].a=0.055f+randf(0.f,0.04f);
        gNeb[i].phase=randf(0.f,TWO_PI);
    }
}

/* Bezier control points (fractions of W,H) for phases 0 and 1 */
static const float PATH[2][4][2]={
    {{0.08f,0.50f},{0.30f,0.20f},{0.60f,0.75f},{0.87f,0.50f}},
    {{0.15f,0.80f},{0.35f,0.30f},{0.55f,0.60f},{0.75f,0.40f}},
};

/* Enter a phase (used by the auto-advance AND the SPACE / 1-2-3 keys) */
static void enterPhase(int ph){
    gPhase=ph; gPhaseTime=0.f; gShip.pathT=0.f; gFade=1.f;
    if(ph<2){
        for(int k=0;k<4;k++){ gShip.cpx[k]=W*PATH[ph][k][0]; gShip.cpy[k]=H*PATH[ph][k][1]; }
        gShip.x=gShip.cpx[0]; gShip.y=gShip.cpy[0]; gShip.shrink=1.f;
    } else {
        gShip.cpx[0]=gShip.x;  gShip.cpy[0]=gShip.y;
        gShip.cpx[1]=W*0.65f;  gShip.cpy[1]=H*0.45f;
        gShip.cpx[2]=W*0.80f;  gShip.cpy[2]=H*0.52f;
        gShip.cpx[3]=gWG.x;    gShip.cpy[3]=gWG.y;
    }
    if(ph==1) gCamShake=2.5f;
}
static void initShip(){
    gShip.angle=0.f; gShip.engineGlow=0.f; gShip.shrink=1.f;
    enterPhase(0);
}
static void initScene(){
    srand(42);
    initStars(); initPlanets(); initAsteroids(); initNebula(); initShip();
}

/* ═══ DRAW ═══
   Background, nebula and stars are drawn in SCREEN space (parallax only),
   so camera motion / zoom-out never reveals empty void beyond the 0..W box. */
static void drawBackground(){
    glDisable(GL_BLEND);
    glColor4f(0.003f,0.001f,0.010f,1.f);
    glBegin(GL_QUADS);
    glVertex2f(0,0);glVertex2f(W,0);glVertex2f(W,H);glVertex2f(0,H);
    glEnd();
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE);
    float bandY=H*0.5f+sinf(gTime*0.07f)*30.f;
    for(int b=0;b<5;b++){
        float t=(float)b/4.f;
        float by=bandY-220.f+b*110.f;
        float a=0.06f*(1.f-fabsf(t-0.5f)*1.8f);
        glBegin(GL_QUADS);
        glColor4f(0.08f,0.04f,0.18f,0.f);
        glVertex2f(0,by);glVertex2f(W,by);
        glColor4f(0.12f,0.06f,0.28f,a);
        glVertex2f(W,by+110.f);glVertex2f(0,by+110.f);
        glEnd();
    }
    float cpx=W*0.5f+sinf(gTime*0.09f)*50.f;
    float cpy=H*0.5f+cosf(gTime*0.07f)*30.f;
    glow(cpx,cpy,280.f,0.15f,0.05f,0.40f,0.06f,8);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
}

static void drawNebula(){
    glBlendFunc(GL_SRC_ALPHA,GL_ONE);
    int layers=gLowQ?2:5;
    for(int i=0;i<NN;i++){
        Nebula& n=gNeb[i];
        float pulse=0.80f+0.20f*sinf(gTime*0.35f+n.phase);
        float bx=wrapf(n.x-gCamOx*0.12f+sinf(gTime*0.10f+i)*12.f,-n.rx*1.8f,W+n.rx*1.8f);
        float by=wrapf(n.y-gCamOy*0.12f+cosf(gTime*0.08f+i)*8.f,-n.ry*1.8f,H+n.ry*1.8f);
        for(int p=0;p<layers;p++){
            float pt=(float)p/layers;
            glColor4f(n.r,n.g,n.b,n.a*(1.f-pt)*pulse*0.65f);
            ellipse(bx,by,n.rx*(1.f+pt*0.7f),n.ry*(1.f+pt*0.7f),40);
        }
        glColor4f(n.r*1.15f,n.g*0.85f,n.b,n.a*0.4f*pulse);
        ellipse(bx+n.rx*0.35f,by-n.ry*0.28f,n.rx*0.50f,n.ry*0.50f,28);
    }
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
}

static void drawStars(){
    glBlendFunc(GL_SRC_ALPHA,GL_ONE);
    for(int i=0;i<NS;i++){
        Star& s=gS[i];
        float tw=0.65f+0.35f*sinf(gTime*1.8f+s.tph);
        float alpha=s.bright*tw;
        /* parallax < 1 => stars scroll slower than the foreground (distant) */
        float px=wrapf(s.x-gCamOx*(0.04f+s.layer*0.22f),0.f,(float)W);
        float py=wrapf(s.y-gCamOy*(0.04f+s.layer*0.22f),0.f,(float)H);
        glColor4f(s.r,s.g,s.b,alpha);
        glPointSize(fmaxf(1.f,s.size*tw));
        glBegin(GL_POINTS);glVertex2f(px,py);glEnd();
        if(!gLowQ && s.bright>0.68f){
            float hr=s.size*3.5f*tw;
            for(int g2=0;g2<3;g2++){
                float gt=(float)g2/3.f;
                glColor4f(s.r,s.g,s.b,alpha*(1.f-gt)*0.28f);
                disc(px,py,hr*(1.f+gt*2.f),10);
            }
            float fl=s.size*7.f*tw*s.bright;
            glColor4f(s.r,s.g,s.b,alpha*0.35f);
            glLineWidth(0.7f);
            glBegin(GL_LINES);
            glVertex2f(px-fl,py);glVertex2f(px+fl,py);
            glVertex2f(px,py-fl);glVertex2f(px,py+fl);
            glEnd();
        }
    }
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glPointSize(1.f); glLineWidth(1.f);
}

/* Saturn-style ring: far half is drawn BEFORE the planet body, near half AFTER,
   so the planet correctly occludes the back of the ring. */
static void planetRing(const Planet& p,bool front){
    float r=p.radius;
    glLineWidth(4.f);
    for(int ri=0;ri<5;ri++){
        float rx=r*(1.42f+ri*0.24f), ry=rx*0.28f;
        glColor4f(p.pr*0.80f,p.pg*0.72f,p.pb*0.55f,0.38f-ri*0.06f);
        glBegin(GL_LINE_STRIP);
        for(int i=0;i<=40;i++){
            float a=(front?PI:0.f)+i*PI/40.f;
            glVertex2f(p.cx+cosf(a)*rx,p.cy+sinf(a)*ry);
        }
        glEnd();
    }
    glLineWidth(1.f);
}

static void drawPlanet(Planet& p){
    float cx=p.cx,cy=p.cy,r=p.radius;
    glBlendFunc(GL_SRC_ALPHA,GL_ONE);
    for(int i=5;i>=0;i--){
        float t=(float)i/5.f;
        glColor4f(p.ar,p.ag,p.ab,0.07f*(1.f-t));
        disc(cx,cy,r*p.atmosSc*(1.f+t*0.55f));
    }
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    if(p.ring) planetRing(p,false);
    int segs=72;
    glBegin(GL_TRIANGLE_FAN);
    glColor4f(fminf(p.pr*1.5f,1.f),fminf(p.pg*1.5f,1.f),fminf(p.pb*1.5f,1.f),1.f);
    glVertex2f(cx-r*0.28f,cy+r*0.30f);
    for(int i=0;i<=segs;i++){
        float a=i*TWO_PI/segs;
        float dx=cosf(a),dy=sinf(a);
        float diff=clampf(0.35f+0.55f*(-0.64f*dx+0.77f*dy),0.15f,1.f);
        glColor4f(p.pr*diff,p.pg*diff,p.pb*diff,1.f);
        glVertex2f(cx+dx*r,cy+dy*r);
    }
    glEnd();
    glPushMatrix();
    glTranslatef(cx,cy,0.f);
    glRotatef(p.rotA*180.f/PI,0.f,0.f,1.f);
    for(int b=0;b<6;b++){
        float by=-r+b*(r*0.38f);
        float bw=sqrtf(fmaxf(0.f,r*r-by*by));
        glColor4f(1.f,1.f,1.f,0.05f+0.035f*(b%2));
        glBegin(GL_QUADS);
        glVertex2f(-bw,by);glVertex2f(bw,by);
        glVertex2f(bw,by+r*0.10f);glVertex2f(-bw,by+r*0.10f);
        glEnd();
    }
    /* surface blobs: hash-based, so the global rand() stream is never touched */
    int sd=(int)(p.blobSeed*10.f);
    for(int s=0;s<5;s++){
        float bx2=lerpf(-r*0.65f,r*0.65f,hash01(sd+s*4));
        float by2=lerpf(-r*0.65f,r*0.65f,hash01(sd+s*4+1));
        float br =lerpf(r*0.07f,r*0.16f,hash01(sd+s*4+2));
        if(bx2*bx2+by2*by2<(r*0.80f)*(r*0.80f)){
            glColor4f(0.f,0.f,0.f,0.18f); disc(bx2,by2,br,14);
            glColor4f(1.f,1.f,1.f,0.06f); disc(bx2-br*0.22f,by2+br*0.22f,br*0.55f,10);
        }
    }
    glPopMatrix();
    if(p.ring) planetRing(p,true);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE);
    glColor4f(1.f,1.f,1.f,0.22f);
    disc(cx-r*0.27f,cy+r*0.30f,r*0.20f,18);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
}

static void drawPlanets(){ for(int i=0;i<NP;i++) drawPlanet(gP[i]); }

static void drawAsteroids(){
    for(int i=0;i<NA;i++){
        Asteroid& a=gA[i];
        glPushMatrix();
        glTranslatef(a.x,a.y,0.f);
        glRotatef(a.rotA*180.f/PI,0.f,0.f,1.f);
        glColor4f(0.38f,0.34f,0.30f,1.f);
        glBegin(GL_TRIANGLE_FAN);
        glVertex2f(0.f,0.f);
        for(int j=0;j<=a.nv;j++){
            int idx=j%a.nv;
            glVertex2f(a.vx2[idx]*a.radius,a.vy2[idx]*a.radius);
        }
        glEnd();
        glColor4f(0.55f,0.50f,0.44f,0.85f);
        glLineWidth(1.1f);
        glBegin(GL_LINE_LOOP);
        for(int j=0;j<a.nv;j++) glVertex2f(a.vx2[j]*a.radius,a.vy2[j]*a.radius);
        glEnd();
        glColor4f(0.22f,0.19f,0.16f,0.55f);
        glBegin(GL_LINES);
        glVertex2f(-a.radius*0.3f,a.radius*0.1f); glVertex2f(a.radius*0.2f,-a.radius*0.2f);
        glVertex2f(a.radius*0.1f,a.radius*0.3f);  glVertex2f(-a.radius*0.1f,0.f);
        glEnd();
        glLineWidth(1.f);
        glPopMatrix();
    }
}

static void drawShip(){
    if(gShip.shrink<0.02f)return;
    float eg=gShip.engineGlow;
    glPushMatrix();
    glTranslatef(gShip.x,gShip.y,0.f);
    glRotatef(gShip.angle*180.f/PI,0.f,0.f,1.f);
    glScalef(gShip.shrink,gShip.shrink,1.f);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE);
    float tl=16.f+eg*65.f;
    glBegin(GL_TRIANGLES);
    glColor4f(0.20f,0.55f,1.0f,0.75f*eg); glVertex2f(-10.f,-4.5f);
    glColor4f(0.05f,0.15f,0.70f,0.0f);    glVertex2f(-10.f-tl,0.f);
    glColor4f(0.20f,0.55f,1.0f,0.75f*eg); glVertex2f(-10.f,4.5f);
    glEnd();
    glBegin(GL_TRIANGLES);
    glColor4f(1.f,1.f,1.f,0.90f*eg);     glVertex2f(-10.f,-1.8f);
    glColor4f(0.80f,0.90f,1.f,0.0f);      glVertex2f(-10.f-tl*0.45f,0.f);
    glColor4f(1.f,1.f,1.f,0.90f*eg);     glVertex2f(-10.f,1.8f);
    glEnd();
    glBegin(GL_TRIANGLES);
    glColor4f(0.55f,0.25f,1.0f,0.55f*eg); glVertex2f(-9.f,-2.8f);
    glColor4f(0.25f,0.05f,0.55f,0.0f);    glVertex2f(-9.f-tl*0.65f,0.f);
    glColor4f(0.55f,0.25f,1.0f,0.55f*eg); glVertex2f(-9.f,2.8f);
    glEnd();
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(0.16f,0.20f,0.30f,1.f);
    glBegin(GL_POLYGON);
    glVertex2f(22.f,0.f);  glVertex2f(13.f,6.5f);
    glVertex2f(-10.f,5.f); glVertex2f(-12.f,2.f);
    glVertex2f(-12.f,-2.f);glVertex2f(-10.f,-5.f);glVertex2f(13.f,-6.5f);
    glEnd();
    glColor4f(0.35f,0.70f,1.0f,0.88f);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(13.f,0.f);
    for(int i=0;i<=20;i++){
        float a=i*PI/20.f-PI*0.5f;
        glVertex2f(13.f+cosf(a)*9.f,sinf(a)*5.5f);
    }
    glEnd();
    glColor4f(1.f,1.f,1.f,0.32f);
    glBegin(GL_TRIANGLE_FAN);
    glVertex2f(16.f,1.5f);
    for(int i=0;i<=8;i++){
        float a=i*PI/8.f-PI*0.5f;
        glVertex2f(16.f+cosf(a)*3.5f,sinf(a)*2.2f+1.5f);
    }
    glEnd();
    glColor4f(0.11f,0.14f,0.24f,1.f);
    glBegin(GL_POLYGON);
    glVertex2f(7.f,6.f);glVertex2f(1.f,22.f);glVertex2f(-6.f,22.f);glVertex2f(-9.f,5.f);
    glEnd();
    glBegin(GL_POLYGON);
    glVertex2f(7.f,-6.f);glVertex2f(1.f,-22.f);glVertex2f(-6.f,-22.f);glVertex2f(-9.f,-5.f);
    glEnd();
    glColor4f(0.13f,0.17f,0.26f,1.f);
    glBegin(GL_QUADS);
    glVertex2f(1.f,22.f);glVertex2f(-6.f,22.f);glVertex2f(-9.f,18.f);glVertex2f(-2.f,18.f);
    glEnd();
    glBegin(GL_QUADS);
    glVertex2f(1.f,-22.f);glVertex2f(-6.f,-22.f);glVertex2f(-9.f,-18.f);glVertex2f(-2.f,-18.f);
    glEnd();
    glBlendFunc(GL_SRC_ALPHA,GL_ONE);
    glColor4f(0.15f,0.85f,1.0f,0.70f);
    glLineWidth(1.4f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(22.f,0.f);glVertex2f(13.f,6.5f);glVertex2f(-10.f,5.f);
    glVertex2f(-12.f,2.f);glVertex2f(-12.f,-2.f);glVertex2f(-10.f,-5.f);glVertex2f(13.f,-6.5f);
    glEnd();
    glColor4f(0.55f,0.28f,1.0f,0.60f);
    glBegin(GL_LINE_LOOP);
    glVertex2f(7.f,6.f);glVertex2f(1.f,22.f);glVertex2f(-6.f,22.f);glVertex2f(-9.f,5.f);
    glEnd();
    glBegin(GL_LINE_LOOP);
    glVertex2f(7.f,-6.f);glVertex2f(1.f,-22.f);glVertex2f(-6.f,-22.f);glVertex2f(-9.f,-5.f);
    glEnd();
    glColor4f(0.70f,0.40f,1.0f,0.45f);
    glBegin(GL_LINES);glVertex2f(-10.f,0.f);glVertex2f(20.f,0.f);glEnd();
    glColor4f(0.35f,0.75f,1.0f,0.65f+eg*0.25f); disc(-10.f,0.f,4.2f,14);
    glColor4f(1.f,1.f,1.f,eg*0.85f);            disc(-10.f,0.f,2.0f,10);
    glColor4f(0.50f,0.25f,1.0f,0.55f+eg*0.25f);
    disc(-8.f,19.f,2.8f,10); disc(-8.f,-19.f,2.8f,10);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glLineWidth(1.f);
    glPopMatrix();
}

static void drawParticles(){
    glBlendFunc(GL_SRC_ALPHA,GL_ONE);
    for(int i=0;i<gNP;i++){
        Particle& p=gParts[i];
        float lt=p.life/p.maxLife;
        if(p.type==0){
            glColor4f(p.r,p.g,p.b,p.a*lt);
            glPointSize(fmaxf(1.f,p.size*lt));   /* size 0 would raise GL_INVALID_VALUE */
            glBegin(GL_POINTS);glVertex2f(p.x,p.y);glEnd();
        } else if(p.type==1){
            glColor4f(p.r,p.g,p.b,p.a*lt);
            glLineWidth(1.3f);
            glBegin(GL_LINES);
            glVertex2f(p.x-p.vx*4.f,p.y-p.vy*4.f);
            glVertex2f(p.x+p.vx*4.f,p.y+p.vy*4.f);
            glEnd();
        } else {
            glColor4f(p.r,p.g,p.b,p.a*lt);
            disc(p.x,p.y,p.size*lt,6);
        }
    }
    glLineWidth(1.f); glPointSize(1.f);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
}

static void drawBlackHole(){
    if(gBH.alpha<0.01f)return;
    float cx=gBH.x,cy=gBH.y,r=gBH.radius,sp=gBH.spin,al=gBH.alpha;
    glBlendFunc(GL_SRC_ALPHA,GL_ONE);
    for(int ri=9;ri>=0;ri--){
        float rr=r*(1.25f+ri*0.42f);
        float ra=0.14f*(1.f-ri/9.f)*al;
        float hue=(float)ri/9.f;
        glColor4f(0.45f+hue*0.35f,0.0f,0.88f-hue*0.25f,ra);
        glBegin(GL_LINE_LOOP);
        for(int s=0;s<80;s++){
            float a=s*TWO_PI/80+sp+ri*0.28f;
            float dr=rr+sinf(a*4.f+gTime*2.2f)*r*0.09f;
            glVertex2f(cx+cosf(a)*dr,cy+sinf(a)*dr*0.82f);
        }
        glEnd();
    }
    for(int d=0;d<64;d++){
        float a=(float)d/64.f*TWO_PI+sp*1.6f;
        float r1=r*1.08f,r2=r*2.85f,t=(float)d/64.f;
        float da=0.48f*al*(0.35f+0.65f*sinf(a*6.f+gTime*3.5f));
        glColor4f(1.0f,0.55f+t*0.30f,0.08f,da);
        glBegin(GL_LINES);
        glVertex2f(cx+cosf(a)*r1,cy+sinf(a)*r1*0.32f);
        glVertex2f(cx+cosf(a+0.14f)*r2,cy+sinf(a+0.14f)*r2*0.32f);
        glEnd();
    }
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(0.f,0.f,0.f,al); disc(cx,cy,r);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE);
    for(int g2=4;g2>=0;g2--){
        float t=(float)g2/4.f;
        glColor4f(0.75f,0.30f,1.0f,(1.f-t)*0.42f*al);
        ring(cx,cy,r*(1.02f+t*0.16f),64);
    }
    glow(cx,cy,r*0.35f,0.9f,0.5f,1.0f,0.15f*al,4);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
}

static void drawWarpGate(){
    if(gWG.alpha<0.01f)return;
    float cx=gWG.x,cy=gWG.y,r=gWG.radius,sp=gWG.spin,al=gWG.alpha;
    glBlendFunc(GL_SRC_ALPHA,GL_ONE);
    for(int ri=0;ri<6;ri++){
        float rr=r*(0.75f+ri*0.22f);
        float ra=(0.18f+gWG.charge*0.25f)*(1.f-ri*0.13f)*al;
        float hue=(float)ri/6.f;
        glColor4f(0.08f+hue*0.55f,0.45f-hue*0.25f,1.f,ra);
        ring(cx,cy,rr,72);
    }
    glLineWidth(1.4f);
    for(int s=0;s<8;s++){
        float a=sp+s*TWO_PI/8.f;
        glColor4f(0.35f,0.70f,1.0f,(0.28f+gWG.charge*0.45f)*al);
        glBegin(GL_LINES);
        glVertex2f(cx+cosf(a)*r*0.28f,cy+sinf(a)*r*0.28f);
        glVertex2f(cx+cosf(a)*r*0.93f,cy+sinf(a)*r*0.93f);
        glEnd();
    }
    glLineWidth(1.f);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(0.f,0.f,0.02f,0.92f*al); disc(cx,cy,r*0.70f);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE);
    for(int s=0;s<55;s++){
        float a=(float)s/55.f*TWO_PI+sp*2.8f;
        float rr=r*0.70f*(0.25f+(float)s/55.f*0.75f);
        glColor4f(0.55f,0.18f,1.0f,0.45f*(1.f-(float)s/55.f)*al);
        glBegin(GL_POINTS);glVertex2f(cx+cosf(a)*rr,cy+sinf(a)*rr);glEnd();
    }
    glow(cx,cy,r*1.15f,0.15f,0.45f,1.f,0.08f+gWG.charge*0.18f,5);  /* stays additive */
    glColor4f(0.0f,1.0f,0.75f,0.85f*al);
    glLineWidth(2.8f);
    glBegin(GL_LINE_STRIP);
    for(int s=0;s<=(int)(72*gWG.charge);s++){
        float a=s*TWO_PI/72-PI*0.5f;
        glVertex2f(cx+cosf(a)*r*1.10f,cy+sinf(a)*r*1.10f);
    }
    glEnd();
    glLineWidth(1.f);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
}

/* Screen-space overlay: always covers the whole view, whatever the camera does */
static void drawHyperspace(){
    if(gPhase!=2)return;
    float t=clampf(gPhaseTime/PHASE_DUR[2],0.f,1.f);
    float alpha=sinf(t*PI);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE);
    glColor4f(0.55f,0.30f,1.0f,alpha*0.40f);
    glBegin(GL_QUADS);glVertex2f(0,0);glVertex2f(W,0);glVertex2f(W,H);glVertex2f(0,H);glEnd();
    /* gate position world -> screen */
    float sx=W*0.5f,sy=H*0.5f;
    float cx=sx+gCamZoom*(gWG.x-sx-gCamOx)+gShkX;
    float cy=sy+gCamZoom*(gWG.y-sy-gCamOy)+gShkY;
    for(int i=0;i<220;i++){
        float a=(float)i/220*TWO_PI;
        float spd=0.40f+0.60f*((float)(i%7)/7.f);
        float len=40.f+280.f*t*spd+hash01(i+gFrame*31)*150.f*t;
        float la=(0.25f+0.70f*t)*alpha*spd;
        glColor4f(0.55f+spd*0.45f,0.65f,1.0f,la);
        glLineWidth(0.4f+t*1.6f);
        glBegin(GL_LINES);
        glVertex2f(cx+cosf(a)*len*0.18f,cy+sinf(a)*len*0.18f);
        glVertex2f(cx+cosf(a)*len,      cy+sinf(a)*len);
        glEnd();
    }
    glLineWidth(1.f);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
}

/* flash used to hide the hard cut between phases */
static void drawFade(){
    if(gFade<0.01f)return;
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glColor4f(0.85f,0.90f,1.0f,gFade*0.9f);
    glBegin(GL_QUADS);glVertex2f(0,0);glVertex2f(W,0);glVertex2f(W,H);glVertex2f(0,H);glEnd();
}

/* HUD: ASCII only (GLUT bitmap fonts cannot draw UTF-8 characters like the em dash) */
static void drawHUD(){
    if(!gShowHUD)return;
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    static const char* pname[]={"PHASE I   - COSMIC CRUISE",
                                 "PHASE II  - BLACK HOLE ENCOUNTER",
                                 "PHASE III - HYPERSPACE ESCAPE"};
    glColor4f(0.4f,0.8f,1.0f,0.70f);
    text(18.f,H-28.f,GLUT_BITMAP_9_BY_15,pname[gPhase]);
    const char* tech[3][3]={
        {"Technique: Parallax Star Field + Gouraud Planet Shading",
         "Particle System: Thruster Exhaust  |  Volumetric Nebula Fog",
         "Bloom: Additive Blending  |  Camera: Bezier Cinematic Path"},
        {"Technique: Black Hole Accretion Disk + Event Horizon Occlusion",
         "Photon Ring Additive Glow  |  Per-vertex Spiral Displacement",
         "Gravity Well: Asteroid & Particle Pull + Swirl"},
        {"Technique: Warp Gate Portal  |  Hyperspace Star Streaks",
         "Radial Stretch Lines from Focal Point  |  Warp Bloom",
         "Full-screen Additive Vignette  |  Particle Burst Trail"},
    };
    glColor4f(0.5f,0.6f,0.7f,0.55f);
    for(int i=0;i<3;i++) text(18.f,26.f-i*16.f,GLUT_BITMAP_8_BY_13,tech[gPhase][2-i]);
    glColor4f(0.7f,0.5f,1.0f,0.85f);
    text(W*0.5f-58.f,H-28.f,GLUT_BITMAP_9_BY_15,"COSMIC ESCAPE");
    char buf[96];
    snprintf(buf,sizeof buf,"FPS %.0f   Speed x%.2f   Quality %s%s",
             gFps,gTimeScale,gLowQ?"LOW":"HIGH",gPaused?"   [PAUSED]":"");
    glColor4f(0.45f,0.85f,0.55f,0.70f);
    text(W-300.f,H-28.f,GLUT_BITMAP_8_BY_13,buf);
    const char* help="SPACE=Next  1/2/3=Phase  P=Pause  +/-=Speed  Q=Quality  H=HUD  S=Shot  F=Fullscreen  ESC=Quit";
    glColor4f(0.35f,0.45f,0.55f,0.55f);
    text(W-8.f*(float)strlen(help)-12.f,10.f,GLUT_BITMAP_8_BY_13,help);
}

/* ═══ UPDATE ═══ */

/* Black-hole gravity: adds a pull + swirl to (vx,vy). Returns true when swallowed. */
static bool pull(float x,float y,float strength,float& vx,float& vy){
    if(gBH.alpha<0.05f)return false;
    float dx=gBH.x-x, dy=gBH.y-y, d2=dx*dx+dy*dy, d=sqrtf(d2)+0.001f;
    if(d<gBH.radius*0.9f)return true;
    float acc=strength*gBH.alpha/(d2+4000.f);
    vx+=(dx/d)*acc+(-dy/d)*acc*0.35f;
    vy+=(dy/d)*acc+( dx/d)*acc*0.35f;
    return false;
}

static void updateParticles(){
    int alive=0;
    for(int i=0;i<gNP;i++){
        Particle& p=gParts[i];
        if(pull(p.x,p.y,500.f,p.vx,p.vy)) p.life=0.f;
        p.x+=p.vx; p.y+=p.vy;
        p.vx*=0.978f; p.vy*=0.978f;
        p.life-=STEP;
        if(p.life>0.f) gParts[alive++]=gParts[i];
    }
    gNP=alive;
}

static void spawnThrust(){
    if(gShip.engineGlow<0.05f || gShip.shrink<0.3f)return;
    float ba=gShip.angle+PI;
    float bvx=cosf(ba)*4.2f, bvy=sinf(ba)*4.2f;
    for(int i=0;i<4;i++){
        float h=randf(0.f,1.f);
        spawnPart(gShip.x,gShip.y,bvx+randf(-0.4f,0.4f),bvy+randf(-0.4f,0.4f),
                  randf(0.35f,0.85f),randf(2.f,5.f),0.25f+h*0.25f,0.60f+h*0.15f,1.f,1.f,0);
    }
    float pa=gShip.angle+PI*0.5f;
    float pfx=cosf(gShip.angle)*(-8.f)+cosf(pa)*19.f+gShip.x;
    float pfy=sinf(gShip.angle)*(-8.f)+sinf(pa)*19.f+gShip.y;
    float pfx2=cosf(gShip.angle)*(-8.f)-cosf(pa)*19.f+gShip.x;
    float pfy2=sinf(gShip.angle)*(-8.f)-sinf(pa)*19.f+gShip.y;
    for(int i=0;i<2;i++){
        spawnPart(pfx, pfy, bvx+randf(-0.2f,0.2f),bvy+randf(-0.2f,0.2f),
                  randf(0.25f,0.55f),randf(1.5f,3.5f),0.5f,0.25f,1.f,0.8f,0);
        spawnPart(pfx2,pfy2,bvx+randf(-0.2f,0.2f),bvy+randf(-0.2f,0.2f),
                  randf(0.25f,0.55f),randf(1.5f,3.5f),0.5f,0.25f,1.f,0.8f,0);
    }
}

static void spawnWarp(){
    if(gShip.shrink<0.3f)return;
    for(int i=0;i<10;i++){
        float a=randf(0.f,TWO_PI),spd=randf(3.f,10.f),h=randf(0.f,1.f);
        spawnPart(gShip.x,gShip.y,cosf(a)*spd,sinf(a)*spd,
                  randf(0.4f,1.2f),randf(1.f,3.f),0.55f+h*0.45f,0.30f+h*0.25f,1.f,1.f,1);
    }
}

static void bezierPt(float t,
    float p0x,float p0y,float p1x,float p1y,
    float p2x,float p2y,float p3x,float p3y,
    float& ox,float& oy)
{
    float u=1.f-t,uu=u*u,uuu=uu*u,tt=t*t,ttt=tt*t;
    ox=uuu*p0x+3*uu*t*p1x+3*u*tt*p2x+ttt*p3x;
    oy=uuu*p0y+3*uu*t*p1y+3*u*tt*p2y+ttt*p3y;
}

static void updateShip(){
    /* path progress is driven by the phase clock, so the ship ALWAYS finishes
       its path exactly when the phase ends (it used to reach only ~70% / 50%) */
    float t=clampf(gPhaseTime/PHASE_DUR[gPhase],0.f,1.f);
    gShip.pathT=(gPhase==2)?powf(t,2.2f):smooth01(t);
    float nx,ny;
    bezierPt(gShip.pathT,
             gShip.cpx[0],gShip.cpy[0],gShip.cpx[1],gShip.cpy[1],
             gShip.cpx[2],gShip.cpy[2],gShip.cpx[3],gShip.cpy[3],nx,ny);
    float dx=nx-gShip.x, dy=ny-gShip.y;
    if(dx*dx+dy*dy>0.01f){
        /* shortest-arc turn (plain lerp spun the long way across +/-PI) */
        float target=atan2f(dy,dx);
        float da=atan2f(sinf(target-gShip.angle),cosf(target-gShip.angle));
        gShip.angle+=da*0.08f;
    }
    gShip.x=nx; gShip.y=ny;
    float spd=sqrtf(dx*dx+dy*dy);
    float tg=clampf(spd/2.5f,0.1f,1.f);
    if(gPhase==2) tg=fmaxf(tg,lerpf(0.4f,1.f,t));
    gShip.engineGlow=lerpf(gShip.engineGlow,tg,0.12f);
    /* in phase III the ship dives into the warp gate and vanishes */
    gShip.shrink=(gPhase==2)?1.f-smooth01((t-0.80f)/0.18f):1.f;
    spawnThrust();
}

static void updateCamera(){
    float tx=gShip.x-W*0.5f+sinf(gTime*0.22f)*35.f;
    float ty=gShip.y-H*0.5f+cosf(gTime*0.17f)*20.f;
    gCamOx=lerpf(gCamOx,tx,0.018f);
    gCamOy=lerpf(gCamOy,ty,0.018f);
    float tz=(gPhase==2)?0.72f:1.f;
    gCamZoom=lerpf(gCamZoom,tz,0.015f);
    gCamShake=fmaxf(0.f,gCamShake-0.08f);
    /* independent X/Y shake (the old code used one value for both = diagonal only) */
    gShkX=(gCamShake>0.f)?randf(-gCamShake,gCamShake):0.f;
    gShkY=(gCamShake>0.f)?randf(-gCamShake,gCamShake):0.f;
}

static void updateBlackHole(){
    if(gPhase==1){ gBH.alpha=lerpf(gBH.alpha,1.f,0.02f); gBH.spin+=0.020f; gBH.radius=60.f+8.f*sinf(gTime*1.4f); }
    else          { gBH.alpha=lerpf(gBH.alpha,0.f,0.03f); }
}

static void updateWarpGate(){
    gWG.spin+=0.016f;
    if(gPhase==2){
        gWG.alpha=lerpf(gWG.alpha,1.f,0.025f);
        gWG.charge=lerpf(gWG.charge,clampf(gPhaseTime/(PHASE_DUR[2]*0.6f),0.f,1.f),0.03f);
        if(rand()%3==0) spawnWarp();
    } else {
        gWG.alpha=lerpf(gWG.alpha,0.f,0.04f);
        gWG.charge=0.f;
    }
}

static void updatePhase(){
    gPhaseTime+=STEP;
    if(gPhaseTime>PHASE_DUR[gPhase]) enterPhase((gPhase+1)%3);
}

static void respawnAsteroid(Asteroid& a){
    float cxw=W*0.5f+gCamOx, cyw=H*0.5f+gCamOy;
    if(rand()%2){ a.x=cxw+(rand()%2?-1.f:1.f)*(ARX-5.f); a.y=randf(cyw-ARY,cyw+ARY); }
    else        { a.y=cyw+(rand()%2?-1.f:1.f)*(ARY-5.f); a.x=randf(cxw-ARX,cxw+ARX); }
    a.bvx=a.vx=randf(-0.35f,0.35f); a.bvy=a.vy=randf(-0.30f,0.30f);
}

static void updateAsteroids(){
    float cxw=W*0.5f+gCamOx, cyw=H*0.5f+gCamOy;
    for(int i=0;i<NA;i++){
        Asteroid& a=gA[i];
        if(pull(a.x,a.y,700.f,a.vx,a.vy)){
            for(int k=0;k<8;k++)
                spawnPart(a.x,a.y,randf(-1.f,1.f),randf(-1.f,1.f),0.6f,5.f,1.f,0.5f,0.1f,0.8f,2);
            respawnAsteroid(a);
            continue;
        }
        /* ease back to the original drift once the pull is gone */
        a.vx=lerpf(a.vx,a.bvx,0.01f); a.vy=lerpf(a.vy,a.bvy,0.01f);
        float sp2=a.vx*a.vx+a.vy*a.vy;
        if(sp2>36.f){ float k=6.f/sqrtf(sp2); a.vx*=k; a.vy*=k; }
        a.x+=a.vx; a.y+=a.vy; a.rotA+=a.rotSpd;
        /* wrap around the camera-centred box, so wrapping is never visible */
        if(a.x<cxw-ARX) a.x+=2.f*ARX;
        else if(a.x>cxw+ARX) a.x-=2.f*ARX;
        if(a.y<cyw-ARY) a.y+=2.f*ARY;
        else if(a.y>cyw+ARY) a.y-=2.f*ARY;
    }
}

static void updateScene(){
    gTime+=STEP; gFrame++;
    gFade=fmaxf(0.f,gFade-0.025f);
    updatePhase(); updateShip(); updateCamera();
    for(int i=0;i<NP;i++) gP[i].rotA+=gP[i].rotSpd;
    updateBlackHole(); updateWarpGate();
    updateAsteroids(); updateParticles();
}

/* ═══ RENDER ═══ */
static void saveScreenshot(){
    int w=gVpW,h=gVpH;
    unsigned char* buf=(unsigned char*)malloc((size_t)w*h*3);
    if(!buf)return;
    glPixelStorei(GL_PACK_ALIGNMENT,1);
    glReadBuffer(GL_BACK);
    glReadPixels(gVpX,gVpY,w,h,GL_RGB,GL_UNSIGNED_BYTE,buf);
    char name[64]; snprintf(name,sizeof name,"cosmic_escape_%03d.ppm",gShots++);
    FILE* f=fopen(name,"wb");
    if(f){
        fprintf(f,"P6\n%d %d\n255\n",w,h);
        for(int y=h-1;y>=0;y--) fwrite(buf+(size_t)y*w*3,1,(size_t)w*3,f);   /* flip rows */
        fclose(f);
        printf("[shot] saved %s\n",name);
    }
    free(buf);
}

static void display(){
    glClear(GL_COLOR_BUFFER_BIT);
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    /* screen-space layers */
    drawBackground();
    drawNebula();
    drawStars();

    /* world-space layers (camera: follow + zoom + shake) */
    float sx=W*0.5f,sy=H*0.5f;
    glPushMatrix();
    glTranslatef(sx+gShkX,sy+gShkY,0.f);
    glScalef(gCamZoom,gCamZoom,1.f);
    glTranslatef(-sx-gCamOx,-sy-gCamOy,0.f);
    drawPlanets();
    drawBlackHole();
    drawAsteroids();
    drawWarpGate();
    drawParticles();
    drawShip();
    glPopMatrix();

    /* screen-space overlays */
    drawHyperspace();
    drawFade();
    drawHUD();
    if(gShotReq){ saveScreenshot(); gShotReq=false; }
    glutSwapBuffers();
}

/* keep the 16:9 aspect (letterbox) instead of stretching the scene */
static void reshape(int w,int h){
    if(h<1)h=1;
    float ar=(float)W/(float)H;
    int vw=w, vh=(int)(w/ar+0.5f);
    if(vh>h){ vh=h; vw=(int)(h*ar+0.5f); }
    gVpX=(w-vw)/2; gVpY=(h-vh)/2; gVpW=vw; gVpH=vh;
    glViewport(gVpX,gVpY,gVpW,gVpH);
    glMatrixMode(GL_PROJECTION);glLoadIdentity();
    gluOrtho2D(0,W,0,H);
    glMatrixMode(GL_MODELVIEW);glLoadIdentity();
}

/* fixed-timestep loop: simulation speed no longer depends on the machine */
static void timerCB(int){
    int now=glutGet(GLUT_ELAPSED_TIME);
    float dt=(now-gLastMs)*0.001f; gLastMs=now;
    if(dt>0.1f)dt=0.1f;
    if(dt>0.f) gFps=lerpf(gFps,1.f/dt,0.1f);
    if(!gPaused){
        gAcc+=dt*gTimeScale;
        int n=0;
        while(gAcc>=STEP && n<8){ updateScene(); gAcc-=STEP; n++; }
        if(n==8) gAcc=0.f;
    }
    glutPostRedisplay();
    glutTimerFunc(10,timerCB,0);
}

static void keyDown(unsigned char key,int,int){
    switch(key){
        case 27:  exit(0);
        case ' ': enterPhase((gPhase+1)%3); break;
        case '1': case '2': case '3': enterPhase(key-'1'); break;
        case 'p': case 'P': gPaused=!gPaused; break;
        case '+': case '=': gTimeScale=fminf(4.f,gTimeScale*1.25f); break;
        case '-': case '_': gTimeScale=fmaxf(0.25f,gTimeScale/1.25f); break;
        case 'h': case 'H': gShowHUD=!gShowHUD; break;
        case 'q': case 'Q': gLowQ=!gLowQ; break;
        case 's': case 'S': gShotReq=true; break;
        case 'f': case 'F':
            gFull=!gFull;
            if(gFull) glutFullScreen();
            else { glutReshapeWindow(W,H); glutPositionWindow(80,60); }
            break;
    }
}

int main(int argc,char** argv){
    printf("=========================================\n");
    printf("  COSMIC ESCAPE - CG Visualization\n");
    printf("  SPACE=Next  1/2/3=Phase  P=Pause  +/-=Speed\n");
    printf("  Q=Quality  H=HUD  S=Screenshot  F=Fullscreen  ESC=Quit\n");
    printf("=========================================\n");
    glutInit(&argc,argv);
    glutInitDisplayMode(GLUT_DOUBLE|GLUT_RGBA);
    glutInitWindowSize(W,H);
    glutInitWindowPosition(80,60);
    glutCreateWindow("COSMIC ESCAPE - Computer Graphics Visualization");
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA,GL_ONE_MINUS_SRC_ALPHA);
    glEnable(GL_POINT_SMOOTH);
    glHint(GL_POINT_SMOOTH_HINT,GL_NICEST);
    glEnable(GL_LINE_SMOOTH);
    glHint(GL_LINE_SMOOTH_HINT,GL_NICEST);
    glClearColor(0.f,0.f,0.f,1.f);
    initScene();
    glutDisplayFunc(display);
    glutReshapeFunc(reshape);
    glutKeyboardFunc(keyDown);
    gLastMs=glutGet(GLUT_ELAPSED_TIME);
    glutTimerFunc(10,timerCB,0);
    printf("[GL] %s | %s\n",(const char*)glGetString(GL_VENDOR),(const char*)glGetString(GL_RENDERER));
    glutMainLoop();
    return 0;
}
