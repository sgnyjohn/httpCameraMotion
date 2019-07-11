# signey julho 2018
# MIT License

# para fazer
# - diferença com contraste
# - tamanho da área atingida
# - área atingida na vertical ou na horizontal
# - área atingida se moveu

import sys
import traceback
import signal
import time

import http.server
import socketserver
import ssl
import io
#from io import StringIO
#from http import cookies
import http.cookies
import mimetypes
import os.path
from imageio import imwrite,imread

# camera
import cv2
from picamera import PiCamera
from picamera.array import PiRGBArray

#import cv2
#from PIL import Image
#import numpy
import threading

import glob

#from skimage.transform import resize

#from scipy.misc import imread,imsave
from scipy.linalg import norm
from scipy import sum, average
def analizaMovSP(img1,img2):
	global e_cron
	# https://gist.github.com/astanin/626356
	def del_gray(arr):
		"# de img colorida-elimina tons cinza = sombras"
		if len(arr.shape) == 3:
			print (arr)
			x = average(arr, -1)
			print(len(x))
			print(len(x[0]))
			r = sum(abs(arr - average(arr, -1)))
			if (r<20):
				arr = [0,0,0]
			return arr
		else:
			return arr
	def to_grayscale(arr):
		"If arr is a color image (3D array), convert it to grayscale (2D array)."
		#tm.ev('gr');
		#print 'gr'
		" roda uma vez... "
		if len(arr.shape) == 3:
			return average(arr, -1)  # average over the last axis (color channels)
		else:
			return arr
	def normalize(arr):
		amin = arr.min()
		rng = arr.max()-amin
		return (arr-amin)*255/rng	
	def compare_images(img1, img2):
		# normalize to compensate for exposure difference
		img1 = normalize(img1)
		img2 = normalize(img2)
		# calculate the difference and its norms
		diff = img1 - img2  # elementwise for scipy arrays
		m_norm = sum(abs(diff))  # Manhattan norm
		z_norm = norm(diff.ravel(), 0)  # Zero norm
		return (m_norm, z_norm)

	#
	e_cron.start('an');
	
	# read images as 2D arrays (convert to grayscale for simplicity)
	if False:
		img1 = img1.astype(float)
		img2 = img2.astype(float)
	else:
		#print len(F1.shape)
		img1 = to_grayscale(img1.astype(float))
		#print len(img1.shape)
		img2 = to_grayscale(img2.astype(float))
	# compare
	n_m, n_0 = compare_images(img1, img2)
	#print "Manhattan norm:", n_m, "/ per pixel:", n_m/img1.size
	#print "Zero norm:", n_0, "/ per pixel:", n_0*1.0/img1.size
	
	e_cron.stop('an');
	
	return n_m/img1.size

def getImageN():
	global capturando,camera,exposAtual,rh
	#continuos
	global c_raw,c_init,monitRunning
	
	dr = time.strftime("%H:%M", time.localtime())
		
	capturando = True
	e_cron.start('cp')
	ti = time.time();
	
	try:
		#inicializar?
		if c_init==0:
			#c_raw = PiRGBArray(camera)
			camera.start_preview()
			time.sleep(0.2)
			c_init = 1
		c_raw = PiRGBArray(camera)
		camera.capture_continuous(c_raw, format="bgr")
		
	except:
		c_init = 0
		lg.erro(sys.exc_info(),"ERRO no capture...")
		return False
		
	e_cron.stop('cp')
	capturando = False
	img = c_raw.array
	
	#ajuste noite/dia ?
	if monitRunning and expos(img):
		c_init = 0

	legImage(img);
	
	return img

def legImage(img):
	" põe legenda na imagem "
	global exposAtual,e_cron,e_imgPorSeg
	# imagens por minuto.
	tc = e_cron.get('cp');
	e_imgPorSeg = '{:.2f}'.format(tc[0]/(e_cron.tempoTotal()))+"ips"
	font = cv2.FONT_HERSHEY_SIMPLEX
	tx = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) \
		+ " " + str(exposAtual) \
		+ " " + e_imgPorSeg \
		+ " " + e_cron.perc()
	cv2.putText(img,tx,(10,rh-10), font, 0.5,(255,255,255),1) #,cv2.LINE_AA)
	
def getImage():
	global capturando,camera,exposAtual,rh,e_cron,c_raw
	
	dr = time.strftime("%H:%M", time.localtime())
		
	capturando = True
	e_cron.start('cp')
	#camera.start_preview()
	#rawCapture = PiRGBArray(camera)
	# allow the camera to warmup
	#time.sleep(0.1)
	# grab an image from the camera
	try:
		print('r='+obj(c_raw)) # camera,size=None,array=(array:[],dtype:uint8)
		c_raw = PiRGBArray(camera)
		camera.capture(c_raw, format="bgr")
	except:
		lg.erro(sys.exc_info(),"ERRO no capture... "+str(e_cron.get('cp')[0]))
		
	e_cron.stop('cp')
	capturando = False
	img = c_raw.array
	
	#ajuste noite/dia ?
	expos(img)
	
	# põe legenda
	legImage(img)

	return img
	
def expos(img):
	" durante dia modo auto, a noite ajuste exposição manualmente "
	" se imagem muito escura entra no modo manual noite "		
	global camera,exposAtual,rw,rh,camSleep,camSleepD,camSleepN,mCor,e_cron
	def Set():
		" seta exposição manual "
		lg.print("camera.exposure_mode = 'off' _speed="+str(exposAtual)+"000 "+str(mCor))
		camera.exposure_mode = 'off'
		#int('vel',camera.shutter_speed)
		camera.framerate=1000/exposAtual
		camera.shutter_speed = exposAtual*1000
		camSleep = camSleepN
		#camera.iso = 800
		
	#calcula a média do valor dos pixels, p/tentar 
		# descobrir luminosidade e regular noite/dia

	e_cron.start('ex');
	if False:
		mc = average(img,-1) #media cor
		mc1 = average(mc,-1) # media colunas
		mCor = average(mc1,-1) # media linhas
	else:
		mCor = average(average(average(img,-1),-1),-1);
	e_cron.stop('ex');

	exIni=100 #inicio
	exInc=100 #incremento
	if mCor<50:
		if exposAtual == -1:
			# inicio anoitecendo
			exposAtual = exIni
			Set()
		else:
			# anoitecendo
			exposAtual += exInc
			Set()
		return True
		

	if mCor>140:
		if exposAtual>exIni:
			#amanhecendo
			exposAtual -= exInc;
			Set()
		else:
			# inicio dia
			lg.print("camera.exposure_mode = 'auto' ="+str(mCor))
			exposAtual = -1
			cameraInit()
			camSleep = camSleepD
		return True
		
	return False
			
def cameraInit():
	global camera,rw,rh,c_raw
	lg.print('init camera');
	if camera!=None:
		camera.close()
		
	camera = PiCamera()
	camera.resolution = (rw, rh)
	camera.start_preview()
	time.sleep(2)
	c_raw = PiRGBArray(camera) #,size=(rw,rh)) 
	lg.print('init camera FIM');

def monitor():
	" tarefa que faz a comparação das imagens geradas pela tarefa captura "
	global imgA,monitRunning,camera,mCor,camSleep,e_imgPorSeg,rw,e_cron
	nv = 0
	nvv = [2,2,2,2,2]
	nd = 10000
	imgA = getImage()
	lg.print("df	md	tempo	mdByte")
	while monitRunning:
		try:
			#image = data.camera()
			#resize(image, (100, 100))
			t = time.time()
			img = getImage()
			df = analizaMovSP(imgA,img);
			# guarda as ultimas nvv.length diferenças
			nvv[nv%len(nvv)] = df
			nv += 1
			md = sum(nvv)/len(nvv)
			#grava dados da img capturada em LOG
			lg.print('{:.2f}'.format(df)
				+"	"+str(int(md*100)/100.0)
				+"	"+str(int((time.time()-t)*1000))
				+"	"+'{:.2f}'.format(mCor)
				+"	"+e_imgPorSeg
				+"	"+e_cron.perc()
				+str(rw)
				#+"	"+str(len(mediaCor1))
				#+"	"+str(len(mediaCor2))
				#+"	2	"+str((mediaCor2))
				#+"	ea	"+str(exposAtual)
				#+"	es	"+str(camera.exposure_speed)
				#+"	em	"+camera.exposure_mode
				#+"	fr	"+str(camera.framerate)
				#+"	iso	"+str(camera.iso)
				#+"	flash	"+camera.flash_mode
			) #n_0*1.0/img1.size/cor
			if df>md*1.6:
				nd += 1
				dr = time.strftime("%Y-%m-%d", time.localtime())
				if not os.path.exists(dr):
					os.mkdir(dr)
				dt = dr+'/'+time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
				print(str(nd)+' '+dt);
				#imsave(dt+'_a_'+str(nd)+'.jpg',imgA)
				#imsave(dt+'_b_'+str(nd)+'.jpg',img)
				imwrite(dt+'_a_'+str(nd)+'.jpg',imgA)
				imwrite(dt+'_b_'+str(nd)+'.jpg',img)
			time.sleep(camSleep/1000)
			imgA = img
		except:
			lg.erro(sys.exc_info(),'ERRO monitor')
	lg.print('finalizando monitor')
	
def startMonitor():
	t = threading.Thread(target=monitor, args=())
	t.daemon = True
	t.start()		

class Handler(http.server.SimpleHTTPRequestHandler):
	
	" servidor http "
	
	#buffer = 1
	#dn = time.strftime("%Y-%m-%d", time.localtime())
	#log_file = open(dn+'/http.log', 'a', buffer)
	#def log_message(self, format, *args):
	#	self.log_file.write("%s - - [%s] %s\n" %
	#		(self.client_address[0],
	#		self.log_date_time_string(),
	#		format%args)
	#	)

	def Dir(self,dr):
		x = [
				(x[0], x[1])
				for x in sorted(
					[
						(fn, os.stat(dr+'/'+fn)) 
						for fn in os.listdir(dr)
					]
					,key = lambda x: x[1].st_mtime
					,reverse = True
				)
		]
		self.h('text/html')
		self.on('<html><head><title>xxx -r0 dir- xxx</title></head>'
			+'<script src="../img.js"></script><body><pre>'
		)
		for i in x:
			self.on(
				i[0]
				+'\t'
				+time.strftime("%Y-%m-%d %H:%M:%S",time.localtime(i[1].st_mtime))
				+'\t'
				+str(i[1].st_size)
			)
		self.on('</pre></body><html>')

	def analizaFoto(self,foto,modelo):
		found = False
		#normalize
		#image = cv2.imread("lenacolor512.tiff", cv2.IMREAD_COLOR)  # uint8 image
		cv2.normalize(foto, foto, 0, 255, cv2.NORM_MINMAX) #, dtype=cv2.CV_32F) # alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
		# converte para gray 
		gray = cv2.cvtColor(foto, cv2.COLOR_BGR2GRAY)
		objects = modelo.detectMultiScale(
			gray,
			scaleFactor=1.1,
			minNeighbors=5,
			minSize=(30, 30),
			flags=cv2.CASCADE_SCALE_IMAGE
		)
		if len(objects) > 0:
			found = True
			#marca a foto
			#cv2.circle(img, center, radius, color[, thickness[, lineType[, shift]]])
			# cor em vez de RGB é BGR
			cv2.circle(foto, (10,10), 20, (0,0,255)) #color[, thickness[, lineType[, shift]]]) → None¶
		# Draw a rectangle around the objects
		for (x, y, w, h) in objects:
			cv2.rectangle(foto, (x, y), (x + w, y + h), (0, 255, 0), 2)
		cv2.circle(foto, (10,10), 20, (0,0,255))
		ret, jpeg = cv2.imencode('.jpg', foto)
		return (jpeg.tobytes(), found)
		#return (io.BytesIO(jpeg.tobytes()), found)

	def dirEventos(self):
		self.h('text/html')
		for i in glob.glob("./10*.jpg"):
			on('<img src="'+i+'">')

	def foto(self):
		" envia slideshow para cliente e se monit parado captura e manda uma "
		global imgA,monitRunning
		t = time.time()
		#https://www.pyimagesearch.com/2015/03/30/accessing-the-raspberry-pi-camera-with-opencv-and-python/
		if monitRunning:
			self.h('image/jpeg')
			image = imgA
			if True:
				jpg = cv2.imencode('.jpg', image)[1]
			else:
				jpg,found = self.analizaFoto(image,modelo)
			self.wfile.write(jpg)			
		elif monitRunning:
			self.h('multipart/x-mixed-replace; boundary=frame')
			image = imgA
			#envia jpg sempre q a imagem se altera
			while True:
				jpg = cv2.imencode('.jpg', image)[1]
				self.wfile.write(b'--frame\r\n')
				self.wfile.write(b'Content-Type: image/jpeg\r\n\r\n')
				self.wfile.write(jpg)
				self.wfile.write(b'\r\n\r\n')
				tf = time.time()
				# espera até 7 segundos caso a imagem não se alterou repete.
				while ((time.time()-tf<7) and (image.all() == imgA.all())):
					time.sleep(0.2)
				if image.all() == imgA.all():
					break
				else:
					time.sleep(0.1)
				image = imgA
		else:
			self.h('image/jpeg')
			image = getImage()
			if True:
				jpg = cv2.imencode('.jpg', image)[1]
			else:
				jpg,found = self.analizaFoto(image,modelo)
			self.wfile.write(jpg)

	def raiz(self):
		self.cookieHeader = self.headers.get('Cookie')
		lg.print('cookie:'+str(self.cookieHeader));
		#self.cookieSet('fig','newton')
		
		self.h('text/html')
		self.on("<html><head><title>r0</title></head>"
			+"<body style=\"width:100%;\">"
			#+"<h1>teste pão</h1>"
			+"<img style=\"image-orientationx: 180deg flip;width:95%;\" src=\"/foto\">"
			+"</body>"
			+"</html>"
		)
		global np
		np += 1

	def mandaArq(self,Aq,pr):
		aq = mStr((www_root+Aq).replace('//','/'))
		#existe ou é diretorio
		if os.path.isdir(aq):
			#index.html existe
			ap = aq+'/index.html'
			if not os.path.isfile(ap):
				self.Dir(aq)
				return
			aq = mStr(ap)
		elif not os.path.isfile(aq):
			self.h('text/html',404);
			self.on('not found "'+Aq+'"')
			return
			
		ex = aq.substrRat('.')
		try:
			mim = mimetypes.types_map['.'+ex]
		except:
			mim = 'application/x-binary'
			
		#print(aq+' e mime: '+mim)
		#aq = mStr(aq);
		if (
				"-py-pyc-php-".find('-'+ex+'.') != -1
				or
				aq.find('..')!=-1
			):
				lg.print("segurança, 'tipo arq' proibido ou '..' .."+aq)
				return
		#envia arq
		self.h(mim)
		f  = open(aq, "rb") 
		bf = '?'
		while len(bf)!=0:
			bf = f.read()
			self.wfile.write(bf)
		#fim
		f.close()
		#print("fim mandaArq");
		
	def do_GET(self):
		dr = mStr(self.path)
		if dr.find('?')!=-1:
			pr = mStr(dr).substrAt('?')
		else:
			pr = ''

		dr = dr.leftAt('?')
		lg.print("dr="+dr+" param="+pr)

		#init cookies
		self.cookie = ''

		try:
			if dr=='/favicon.ico':
				return;
			#print("path:"+dr)
			if dr=='xx/':
				self.raiz()
			elif dr=='/setExposicao':
				global exposHttp
				exposHttp = float(pr);
			elif dr=='/foto':
				self.foto()
			elif dr=='/stop':
				self.h('text/html',200);
				self.on('stop')
				lg.print("sair solicitado pela web")
				sair()				
			elif dr=='/dir':
				self.dirEventos()
			else:
				self.mandaArq(dr,pr)

		except Exception:
			lg.erro(sys.exc_info(),'ERRO handler HTTTP')
		
		return

	def cookieSet(self,n,v):
		C = http.cookies.SimpleCookie()
		C[n] = v
		self.cookie += C.output(header='')+'; '

	def h(self,mime,cod=200):
		self.send_response(cod)
		#self.send_header('Content-Disposition', 'attachment; filename=test.xlsx')
		if mime.startswith('text/') and mime.find(';')==-1:
			mime += '; charset='+charset
		self.send_header('Content-type',mime)
		#set cook
		if self.cookie != '':
			self.send_header('Set-Cookie', self.cookie)
			self.cookie = ''
		#Content-Security-Policy:
		if False: 
			csp = "default-src 'self'; img-src https://*; child-src 'none'";
			csp = "default-src *; style-src 'self' 'unsafe-inline'; script-src 'self' 'unsafe-inline' 'unsafe-eval' http://www.google.com";
			self.send_header('Content-Security-Policy',csp)
		#fim
		self.end_headers()

	def on(self,s):
		self.wfile.write(bytes(s+'\n', charset))

# biblioteca
class mStr(str):
	def leftAt(self,sf):
		p = self.find(sf)
		if p == -1:
			return self
		return self[:p]

	def substrRat(self,sf):
		p = self.rfind(sf)
		if p == -1:
			return ''
		return self[p+len(sf):]

	def substrAt(self,sf):
		p = self.find(sf)
		if p == -1:
			return ''
		return self[p+len(sf):]


def obj(o):
	return str(dir(o))
	#r = ''
	#for i in o:
	#	r += str(i)+' tp('+str(type(o[i]))+')=('+str(o[i])+')\n'
	#return r;

class cronometro(object):
	" cria vários cronometros reativáveis "
	" para medir desempenho de partes do código "
	#y_dict = {1: 'apple', 2: 'ball'}
	def perc(self):
		" retorna string com percentuais dos cronometros "
		r = ''
		for i in self.v:
			v = self.v[i]
			r += i+"%"+'{:.1f}'.format(v[1]/self.tempoTotal()*100)+' '
		return r
	def tempoTotal(self):
		return time.time()-self.tt
	def start(self,s):
		v = None
		try:
			v = self.v[s]
		except:
			v = [0,0,-1]
			self.v[s] = v
		v[2] = time.time();
	def stop(self,s):
		v = self.v[s]
		v[0] += 1;
		v[1] += time.time()-v[2];
		v[2] = -1
	def get(self,s):
		return self.v[s]
	def txt(self):
		r = '\n==> '+self.tit+'\tnv\tms\tmsMD'
		for i in self.v:
			r += ("\n"+i
				+"\t"+str(self.v[i][0])
				+"\t"+str(int(self.v[i][1]*1000))
				+"\t"+str(int(self.v[i][1]*1000/self.v[i][0]))
			)
		return r
	def __init__(self,a):
		self.tit = a
		self.tt = time.time()
		self.v = {}

class log(object):
	def __init__(self,path):
		self.path = path
		self.file = open(self.path,'a')
	def rotateDay(self,tm=-1):
		st = os.stat(self.path)
		df = time.strftime("%Y-%m-%d", time.localtime(st.st_mtime))
		dn = time.strftime("%Y-%m-%d", time.localtime())
		if df != dn and ( tm==-1 or st.st_size > tm ):
			self.file.close()
			os.rename(self.path,self.path+'-'+df)
			self.file = open(self.path,'a') 
	def print(self,ln):
		dt = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
		print('lg\t'+dt+'\t'+ln)
		if ln.find('\r')!=-1: ln = ln.replace('\r','')
		if ln.find('\n')!=-1: ln = ln.replace('\n','\\n')
		if ln.find('\t')!=-1: ln = ln.replace('\t','\\t')
		self.file.write(dt+'\t'+ln+'\n')
		self.file.flush()
	def close(self):
		self.file.close()
	def erro(self,err,tx='Unexpected ERROR'):
		tr = ''.join(traceback.format_tb(err[2]))
		# (type, value, traceback)
		self.print('==============>> '+tx)
		self.print('\tTYPE:'+str(err[0]))
		self.print('\tVALUE:'+str(err[1]))
		self.print('\tTRACE: '+tr)
		#err[2].print_exc(file=self.file)
		#err[2].print_exc(file=sys.stdout)

class GracefulKiller:
  kill_now = False
  def __init__(self):
    signal.signal(signal.SIGINT, self.exit_gracefully)
    signal.signal(signal.SIGTERM, self.exit_gracefully)

  def exit_gracefully(self,signum, frame):
    self.kill_now = True

def sairKill():
	lg.print("saindo via KILL")
	sair()

def sair():
	global camera,monitRunning,killer
	monitRunning = False
	camera.close()
	if capturando:
		lg.print("aguardando fim captura...")
		while capturando:
			print(".")
			monitRunning = False
			time.sleep(0.2)
		lg.print('fim captura')
	httpd.server_close()
	lg.print('sair(): saida normal...')
	killer.kill_now = True

def startHttp():
	global httpd
	while monitRunning:
		httpd.handle_request()
	#httpd.serve_forever()	
		
#==================================================================
# main inicio principal
#==================================================================

try:	
	#logs
	dn = time.strftime("%Y-%m-%d", time.localtime())
	if not os.path.exists(dn):
		os.mkdir(dn)
	lg = log(dn+'/prg.log')
	#lg.rotateDay()
	lg.print('inicio log...')

	#camera
	#camera = PiCamera(resolution=(1280, 720), framerate=30)
	#rh=240;rw=320
	#rh=480;rw=640
	rh=768;rw=1024
	#rh=1920;rw=2560

	#cap continuo 
	c_raw=None
	c_init=0

	camera=None
	cameraInit()

	#tabela de horario claridade
	mCor = 0.0 #media cor
	# DIURNO
	camSleepD=20
	# NOTURNO
	camSleepN=20
	# var exec
	camSleep=camSleepD
	exposAtual = -1
	
	
	#estatisticas
	e_imgPorSeg = 0
	e_cron = cronometro('desempenho codigo');

	#bug ao abortar durante capture
	# var controle de capture ocorrendo
	capturando = False
	
	#load mimetypes
	mimetypes.init()
	#print("arqs mime: "+str(mimetypes.knownfiles))

	monitRunning = True
	imgA = False
	startMonitor()
	
	#time.sleep(60);
	

	#inicia SERVIDOR HTTP
	www_root = '.'
	charset = 'utf-8'
	port = 8043
	lg.print('Server listening on port '+str(port)+'...')
	httpd = socketserver.TCPServer(('', port), Handler)
	# gerar chave 
	# openssl req -new -x509 -days 4000 -nodes -out chave.crt -keyout chave.key
	httpd.socket = ssl.wrap_socket (httpd.socket, certfile='chave.crt', keyfile='chave.key', server_side=True)
	#httpd.serve_forever()
	t = threading.Thread(target=startHttp, args=())
	t.daemon = True
	t.start()	
	
	#aguarda ser MORTO
	killer = GracefulKiller()
	while True:
		time.sleep(3) #segundos
		if killer.kill_now:
			break
	#fecha tudo...
	sairKill()	
		
except	KeyboardInterrupt:
	lg.print("saida solicitada pelo teclado!")
	sair()

except:
	print (sys.exc_info(),'ERRO prg principal')
	lg.erro(sys.exc_info(),'ERRO prg principal')

finally:
	camera.close()
	lg.print("finally FIM...")

lg.close()
