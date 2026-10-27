"""CAPYBARA BOUNCE RUN 🐹
Fullscreen endless runner: Pygame + OpenCV HSV + MediaPipe Tasks.

Install:
  py -3.13 -m pip install pygame opencv-python mediapipe numpy
Run:
  py -3.13 capybara_bounce_run.py
Controls: fist = jump, raise hand = high jump, SPACE = manual jump,
R = restart after game over, ESC/Q = quit.
"""
import math, time, urllib.request
from pathlib import Path
import cv2
import mediapipe as mp
import numpy as np
import pygame

FPS = 60
GROUND_RATIO = 0.80
ROOT = Path(__file__).resolve().parent
ASSET_DIR = ROOT / "assets"
MODEL_PATH = ROOT / "hand_landmarker.task"
MODEL_URL = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
HSV_LOWER = np.array([0, 25, 50], dtype=np.uint8)
HSV_UPPER = np.array([25, 200, 255], dtype=np.uint8)
CONNECTIONS = [(0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(5,9),(9,10),(10,11),(11,12),(9,13),(13,14),(14,15),(15,16),(13,17),(17,18),(18,19),(19,20),(0,17)]


def ensure_model():
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 100_000:
        return
    print("Mengunduh model MediaPipe...")
    try:
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(f"Model gagal diunduh. Periksa internet. Detail: {exc}")


def make_landmarker():
    ensure_model()
    options = mp.tasks.vision.HandLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_hands=1,
        min_hand_detection_confidence=0.55,
        min_hand_presence_confidence=0.50,
        min_tracking_confidence=0.50,
    )
    return mp.tasks.vision.HandLandmarker.create_from_options(options)


def load_sprite(name, size):
    path = ASSET_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Aset tidak ditemukan: {path}")
    image = pygame.image.load(str(path)).convert_alpha()
    return pygame.transform.smoothscale(image, size)


def text(screen, value, font, color, pos):
    screen.blit(font.render(value, True, color), pos)


def round_rect(screen, color, rect, radius=14):
    pygame.draw.rect(screen, color, rect, border_radius=radius)


def cloud(screen, x, y, s):
    c=(255,255,255)
    pygame.draw.circle(screen,c,(int(x),int(y)),max(2,int(20*s)))
    pygame.draw.circle(screen,c,(int(x+20*s),int(y-8*s)),max(2,int(25*s)))
    pygame.draw.circle(screen,c,(int(x+44*s),int(y)),max(2,int(19*s)))
    pygame.draw.rect(screen,c,(int(x-2*s),int(y),int(49*s),int(14*s)))


def background(screen, scroll, w, h, ground, s):
    screen.fill((118,205,255))
    pygame.draw.circle(screen,(255,229,126),(int(w*.84),int(h*.13)),max(10,int(34*s)))
    cloud(screen,w*.10,h*.15,s)
    cloud(screen,w*.42,h*.20,.8*s)
    cloud(screen,w*.67,h*.12,.75*s)
    pygame.draw.polygon(screen,(151,204,132),[(0,int(h*.56)),(int(w*.14),int(h*.36)),(int(w*.29),int(h*.56)),(int(w*.45),int(h*.40)),(int(w*.65),int(h*.57)),(int(w*.84),int(h*.38)),(w,int(h*.55)),(w,ground),(0,ground)])
    pygame.draw.polygon(screen,(111,179,111),[(0,int(h*.65)),(int(w*.17),int(h*.48)),(int(w*.33),int(h*.65)),(int(w*.51),int(h*.50)),(int(w*.71),int(h*.66)),(int(w*.89),int(h*.49)),(w,int(h*.61)),(w,ground),(0,ground)])
    fy=int(ground-82*s); spacing=max(28,int(100*s)); off=int(scroll*.45)%spacing
    for x in range(-spacing-off,w+spacing,spacing):
        pygame.draw.rect(screen,(167,119,78),(x,fy,max(4,int(8*s)),int(62*s)),border_radius=3)
    pygame.draw.rect(screen,(196,149,99),(0,fy+int(17*s),w,max(3,int(7*s))))
    pygame.draw.rect(screen,(196,149,99),(0,fy+int(42*s),w,max(3,int(7*s))))
    pygame.draw.rect(screen,(117,181,73),(0,ground,w,h-ground))
    pygame.draw.rect(screen,(86,142,55),(0,ground,w,max(4,int(8*s))))
    pygame.draw.rect(screen,(161,111,72),(0,ground+int(17*s),w,max(0,h-ground-int(17*s))))
    spacing=max(18,int(52*s)); off=int(scroll)%spacing
    for x in range(-spacing-off,w+spacing,spacing):
        pygame.draw.line(screen,(72,134,47),(x,ground+int(2*s)),(x+int(5*s),ground-int(5*s)),max(1,int(2*s)))
        pygame.draw.line(screen,(72,134,47),(x+int(5*s),ground-int(5*s)),(x+int(10*s),ground+int(2*s)),max(1,int(2*s)))


def draw_obstacle(screen, obs, ground, s):
    x=int(obs['x'])
    if obs['kind']=='crate':
        y=ground-int(47*s); ww=int(42*s); hh=int(47*s)
        pygame.draw.rect(screen,(139,80,43),(x,y,ww,hh),border_radius=max(2,int(4*s)))
        pygame.draw.rect(screen,(207,143,81),(x+int(4*s),y+int(4*s),int(34*s),int(39*s)),border_radius=3)
        pygame.draw.line(screen,(125,72,39),(x+int(6*s),y+int(6*s)),(x+int(36*s),y+int(40*s)),max(2,int(4*s)))
        pygame.draw.line(screen,(125,72,39),(x+int(36*s),y+int(6*s)),(x+int(6*s),y+int(40*s)),max(2,int(4*s)))
    else:
        pygame.draw.polygon(screen,(76,77,87),[(x,ground),(x+int(22*s),ground-int(43*s)),(x+int(44*s),ground)])
        pygame.draw.polygon(screen,(235,235,224),[(x+int(8*s),ground-int(3*s)),(x+int(22*s),ground-int(34*s)),(x+int(29*s),ground-int(3*s))])


def gesture_of(lm, ratio):
    if not lm or ratio < 0.025:
        return 'none'
    extended=sum(lm[t].y < lm[p].y for t,p in [(8,6),(12,10),(16,14),(20,18)])
    if extended >= 3: return 'open'
    if extended <= 1: return 'fist'
    return 'other'


def detect(frame, landmarker, timestamp):
    rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
    image=mp.Image(image_format=mp.ImageFormat.SRGB,data=rgb)
    result=landmarker.detect_for_video(image,timestamp)
    hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,HSV_LOWER,HSV_UPPER)
    kernel=np.ones((3,3),np.uint8)
    mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,kernel)
    mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,kernel)
    gesture='none'; center_y=None; hand_center=None; ratio=0.0
    if result.hand_landmarks:
        lm=result.hand_landmarks[0]; h,w=frame.shape[:2]
        xs=[p.x*w for p in lm]; ys=[p.y*h for p in lm]
        x1=max(0,int(min(xs)-18)); y1=max(0,int(min(ys)-18))
        x2=min(w,int(max(xs)+18)); y2=min(h,int(max(ys)+18))
        roi=mask[y1:y2,x1:x2]
        if roi.size: ratio=cv2.countNonZero(roi)/roi.size
        gesture=gesture_of(lm,ratio); center_y=float(np.mean(ys))
        clean=roi
        if clean.size:
            yy,xx=np.where(clean>0)
            if len(xx)>80: hand_center=(int(np.mean(xx))+x1,int(np.mean(yy))+y1)
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,220,255),2)
        points=[(int(p.x*w),int(p.y*h)) for p in lm]
        for a,b in CONNECTIONS: cv2.line(frame,points[a],points[b],(50,255,70),2,cv2.LINE_AA)
        for pt in points: cv2.circle(frame,pt,4,(30,60,255),-1,cv2.LINE_AA)
        if hand_center: cv2.circle(frame,hand_center,7,(255,0,255),-1)
        cv2.putText(frame,f'Gestur: {gesture} | HSV: {ratio:.2f}',(10,27),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
    else:
        cv2.putText(frame,'Arahkan tangan ke kamera',(10,27),cv2.FONT_HERSHEY_SIMPLEX,.55,(255,255,255),2)
    return gesture,center_y,mask,hand_center


def cv_surface(frame, size, is_mask=False):
    resized=cv2.resize(frame,size,interpolation=cv2.INTER_AREA)
    rgb=cv2.cvtColor(resized,cv2.COLOR_GRAY2RGB) if is_mask else cv2.cvtColor(resized,cv2.COLOR_BGR2RGB)
    return pygame.surfarray.make_surface(rgb.swapaxes(0,1))


def main():
    pygame.init()
    pygame.display.set_caption('Capybara Bounce Run')
    screen=pygame.display.set_mode((0,0),pygame.FULLSCREEN)
    w,h=screen.get_size(); s=min(w/1100,h/650); ground=int(h*GROUND_RATIO)
    print(f'Resolusi fullscreen: {w} x {h}')
    clock=pygame.time.Clock()
    title=pygame.font.SysFont('arial',max(20,int(34*s)),bold=True)
    font=pygame.font.SysFont('arial',max(15,int(22*s)),bold=True)
    small=pygame.font.SysFont('arial',max(11,int(14*s)))
    sprite_size=(max(60,int(120*s)),max(60,int(120*s)))
    sprites={name:load_sprite(file,sprite_size) for name,file in {
        'idle':'capybara_idle.png','run1':'capybara_run1.png','run2':'capybara_run2.png','jump':'capybara_jump.png','fall':'capybara_fall.png'}.items()}
    camera=None; landmarker=None
    try:
        camera=cv2.VideoCapture(0)
        if not camera.isOpened(): raise RuntimeError('Kamera tidak dapat dibuka. Tutup aplikasi lain yang memakai kamera.')
        camera.set(cv2.CAP_PROP_FRAME_WIDTH,640); camera.set(cv2.CAP_PROP_FRAME_HEIGHT,480)
        landmarker=make_landmarker()
    except Exception as exc:
        if camera: camera.release()
        pygame.quit(); raise SystemExit(f'Gagal memulai kamera/model:\n{exc}')

    player_x=int(w*.13); player_y=ground-int(90*s); player_vy=0.0
    gravity=.82*s; jump_power=-15*s; high_power=-19*s; on_ground=True
    obstacles=[]; spawn_timer=0; score=0; best=0; lives=3; speed=6*s; scroll=0; run_frame=0
    game_over=False; last_gesture='none'; prev_gesture='none'; last_jump=0; last_high=0; timestamp=0
    camera_surface=None; mask_surface=None; hand_center=None
    jump_requested=False; high_requested=False
    panel_w=max(175,min(int(w*.19),int(235*s))); panel_h=max(195,min(int(h*.43),int(285*s)))
    panel_x=w-panel_w-int(14*s); panel_y=int(14*s)
    try:
        while True:
            dt=clock.tick(FPS); timestamp+=max(1,dt); now=time.time()
            for event in pygame.event.get():
                if event.type==pygame.QUIT: return
                if event.type==pygame.KEYDOWN:
                    if event.key in (pygame.K_ESCAPE,pygame.K_q): return
                    if event.key==pygame.K_SPACE and not game_over: jump_requested=True
                    if event.key==pygame.K_r and game_over:
                        player_y=ground-int(90*s); player_vy=0; on_ground=True; obstacles.clear(); spawn_timer=0; score=0; lives=3; speed=6*s; game_over=False; prev_gesture='none'; last_jump=0
            ok,frame=camera.read()
            high_requested=False
            if ok:
                frame=cv2.flip(frame,1)
                gesture,cy,mask,hand_center=detect(frame,landmarker,timestamp)
                last_gesture=gesture
                if gesture=='fist' and prev_gesture!='fist' and now-last_jump>.25:
                    jump_requested=True; last_jump=now
                if cy is not None and cy<frame.shape[0]*.25 and now-last_high>.9:
                    high_requested=True; jump_requested=True; last_high=now
                prev_gesture=gesture
                camera_surface=cv_surface(frame,(320,240))
                mask_surface=cv_surface(mask,(320,240),True)
            if not game_over:
                if jump_requested and on_ground:
                    player_vy=high_power if high_requested else jump_power; on_ground=False
                player_vy+=gravity; player_y+=player_vy
                floor=ground-int(90*s)
                if player_y>=floor: player_y=floor; player_vy=0; on_ground=True
                speed=min(11.5*s,(6+score/140)*s); scroll+=speed; run_frame+=1; spawn_timer+=1
                if spawn_timer>max(42,int(94-(speed/max(s,.1))*3)):
                    obstacles.append({'x':w+int(20*s),'kind':'spike' if np.random.random()<.58 else 'crate','passed':False,'hit':False}); spawn_timer=0
                # Sprite collision box slightly inset from transparent canvas.
                player_rect=pygame.Rect(int(player_x+20*s),int(player_y+28*s),int(78*s),int(68*s))
                for obs in obstacles:
                    obs['x']-=speed; ox=obs['x']; oh=43 if obs['kind']=='spike' else 47; oy=ground-int(oh*s)
                    obs_rect=pygame.Rect(int(ox+5*s),int(oy+5*s),int(34*s),int(40*s))
                    if player_rect.colliderect(obs_rect) and not obs['hit']:
                        obs['hit']=True; lives-=1
                        if lives<=0: game_over=True
                    if not obs['passed'] and ox<player_x-20*s: obs['passed']=True; score+=1
                obstacles=[o for o in obstacles if o['x']>-80*s]; best=max(best,score)
            # Render game
            background(screen,scroll,w,h,ground,s)
            for obs in obstacles: draw_obstacle(screen,obs,ground,s)
            sprite_key='fall' if game_over else ('jump' if not on_ground and player_vy<0 else ('run1' if int(run_frame/8)%2==0 else 'run2'))
            if on_ground and run_frame<1: sprite_key='idle'
            screen.blit(sprites[sprite_key],(int(player_x),int(player_y)))
            round_rect(screen,(35,73,104),(int(20*s),int(18*s),int(280*s),int(92*s)),max(5,int(18*s)))
            text(screen,f'Skor: {score}',title,(255,255,255),(int(38*s),int(27*s)))
            text(screen,f'Terbaik: {best}',font,(255,230,135),(int(40*s),int(72*s)))
            heart_w=int(170*s)
            round_rect(screen,(35,73,104),(w-heart_w-int(18*s),int(18*s),heart_w,int(52*s)),max(5,int(16*s)))
            text(screen,'♥ '*lives+'♡ '*(3-lives),font,(255,104,126),(w-heart_w-int(5*s),int(28*s)))
            # Embedded camera/mask preview. No separate cv2.imshow windows.
            px,py,pw,ph=panel_x,panel_y,panel_w,panel_h
            round_rect(screen,(30,43,63),(px,py,pw,ph),max(6,int(12*s)))
            text(screen,'KAMERA + LANDMARK',small,(255,255,255),(px+int(7*s),py+int(5*s)))
            inner_w=pw-int(14*s); cam_h=int(ph*.43); cam_y=py+int(25*s)
            if camera_surface:
                screen.blit(pygame.transform.smoothscale(camera_surface,(inner_w,cam_h)),(px+int(7*s),cam_y))
                pygame.draw.rect(screen,(255,255,255),(px+int(7*s),cam_y,inner_w,cam_h),1)
            mask_y=cam_y+cam_h+int(24*s); text(screen,'MASK HSV',small,(255,255,255),(px+int(7*s),mask_y-int(20*s)))
            mask_h=max(35,py+ph-mask_y-int(7*s))
            if mask_surface:
                screen.blit(pygame.transform.smoothscale(mask_surface,(inner_w,mask_h)),(px+int(7*s),mask_y))
                pygame.draw.rect(screen,(255,255,255),(px+int(7*s),mask_y,inner_w,mask_h),1)
            bottom_h=max(35,int(54*s)); bottom_y=h-bottom_h-int(14*s)
            round_rect(screen,(255,248,230),(int(20*s),bottom_y,w-int(40*s),bottom_h),max(5,int(14*s)))
            labels={'fist':'KEPAL = LOMPAT','open':'TELAPAK TERBUKA = LARI','other':'ATUR GESTUR TANGAN','none':'TUNJUKKAN TANGAN KE KAMERA'}
            text(screen,labels.get(last_gesture,'TUNJUKKAN TANGAN'),font,(89,73,55),(int(34*s),bottom_y+int(10*s)))
            text(screen,'Angkat tangan = lompat tinggi | SPACE = lompat | ESC = keluar',small,(89,73,55),(int(w*.43),bottom_y+int(17*s)))
            if game_over:
                overlay=pygame.Surface((w,h),pygame.SRCALPHA); overlay.fill((35,35,45,175)); screen.blit(overlay,(0,0))
                bw=int(min(w*.75,500*s)); bh=int(min(h*.5,220*s)); bx=(w-bw)//2; by=(h-bh)//2
                round_rect(screen,(255,246,224),(bx,by,bw,bh),max(8,int(22*s)))
                text(screen,'YAHH, CAPYBARA JATUH!',title,(116,73,47),(bx+int(25*s),by+int(35*s)))
                text(screen,f'Skor kamu: {score}',font,(116,73,47),(bx+int(30*s),by+int(95*s)))
                text(screen,'Tekan R untuk main lagi',font,(116,73,47),(bx+int(30*s),by+int(135*s)))
                text(screen,'ESC untuk keluar',small,(116,73,47),(bx+int(30*s),by+int(175*s)))
            pygame.display.flip()
            jump_requested=False
    finally:
        camera.release()
        landmarker.close()
        pygame.quit()

if __name__=='__main__':
    main()
