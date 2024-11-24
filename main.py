especular = True

# -----------------------------------x
import glfw
from OpenGL.GL import *
import OpenGL.GL.shaders
import numpy as np
import glm
import math
from PIL import Image
from objeto import Objeto
# -----------------------------------

glfw.init()
glfw.window_hint(glfw.VISIBLE, glfw.FALSE);
largura = 1920
altura = 1080
window = glfw.create_window(largura, altura, "Iluminação", None, None)
glfw.make_context_current(window)


# ----------------------------------------

vertex_code = """
        attribute vec3 position;
        attribute vec2 texture_coord;
        attribute vec3 normals;
        
       
        varying vec2 out_texture;
        varying vec3 out_fragPos; //posicao do fragmento (i.e., posicao na superficie onde a iluminacao sera calculada)
        varying vec3 out_normal;
                
        uniform mat4 model;
        uniform mat4 view;
        uniform mat4 projection;        
        
        void main(){
            gl_Position = projection * view * model * vec4(position,1.0);
            out_texture = vec2(texture_coord);
            out_fragPos = vec3(  model * vec4(position, 1.0));
            out_normal = vec3( model *vec4(normals, 1.0));            
        }
        """


# ------------------------------------------

if not especular:

    fragment_code = """
    
            // parametro com a cor da(s) fonte(s) de iluminacao
            uniform vec3 lightPos; // define coordenadas de posicao da luz
            vec3 lightColor = vec3(1.0, 1.0, 1.0);
            
            // parametros da iluminacao ambiente e difusa
            uniform float ka; // coeficiente de reflexao ambiente
            uniform float kd; // coeficiente de reflexao difusa

    
            // parametros recebidos do vertex shader
            varying vec2 out_texture; // recebido do vertex shader
            varying vec3 out_normal; // recebido do vertex shader
            varying vec3 out_fragPos; // recebido do vertex shader
            uniform sampler2D samplerTexture;
            
            void main(){
            
                // calculando reflexao ambiente
                vec3 ambient = ka * lightColor;             
            
                // calculando reflexao difusa
                vec3 norm = normalize(out_normal); // normaliza vetores perpendiculares
                vec3 lightDir = normalize(lightPos - out_fragPos); // direcao da luz
                float diff = max(dot(norm, lightDir), 0.0); // verifica limite angular (entre 0 e 90)
                vec3 diffuse = kd * diff * lightColor; // iluminacao difusa
                
               
                
                // aplicando o modelo de iluminacao
                vec4 texture = texture2D(samplerTexture, out_texture);
                vec4 result = vec4((ambient + diffuse),1.0) * texture; // aplica iluminacao
                gl_FragColor = result;
    
            }
            """

else:
    
    fragment_code = """
    
            // parametro com a cor da(s) fonte(s) de iluminacao
            uniform vec3 lightPos; // define coordenadas de posicao da luz
            vec3 lightColor = vec3(1.0, 1.0, 1.0);
            
            // parametros da iluminacao ambiente e difusa
            uniform float ka; // coeficiente de reflexao ambiente
            uniform float kd; // coeficiente de reflexao difusa
            
            // parametros da iluminacao especular
            uniform vec3 viewPos; // define coordenadas com a posicao da camera/observador
            uniform float ks; // coeficiente de reflexao especular
            uniform float ns; // expoente de reflexao especular
            
    
    
            // parametros recebidos do vertex shader
            varying vec2 out_texture; // recebido do vertex shader
            varying vec3 out_normal; // recebido do vertex shader
            varying vec3 out_fragPos; // recebido do vertex shader
            uniform sampler2D samplerTexture;
            
            
            
            void main(){
            
                // calculando reflexao ambiente
                vec3 ambient = ka * lightColor;             
            
                // calculando reflexao difusa
                vec3 norm = normalize(out_normal); // normaliza vetores perpendiculares
                vec3 lightDir = normalize(lightPos - out_fragPos); // direcao da luz
                float diff = max(dot(norm, lightDir), 0.0); // verifica limite angular (entre 0 e 90)
                vec3 diffuse = kd * diff * lightColor; // iluminacao difusa
                
                // calculando reflexao especular
                vec3 viewDir = normalize(viewPos - out_fragPos); // direcao do observador/camera
                vec3 reflectDir = normalize(reflect(-lightDir, norm)); // direcao da reflexao
                float spec = pow(max(dot(viewDir, reflectDir), 0.0), ns);
                
                vec3 specular = ks * spec * lightColor;             
                
                // aplicando o modelo de iluminacao
                vec4 texture = texture2D(samplerTexture, out_texture);
                vec4 result = vec4((ambient + diffuse + specular),1.0) * texture; // aplica iluminacao
                gl_FragColor = result;
    
            }
            """

            # -------------------------------------------------------------------------------

# Request a program and shader slots from GPU
program  = glCreateProgram()
vertex   = glCreateShader(GL_VERTEX_SHADER)
fragment = glCreateShader(GL_FRAGMENT_SHADER)


# Set shaders source
glShaderSource(vertex, vertex_code)
glShaderSource(fragment, fragment_code)

# Compile shaders
glCompileShader(vertex)
if not glGetShaderiv(vertex, GL_COMPILE_STATUS):
    error = glGetShaderInfoLog(vertex).decode()
    print(error)
    raise RuntimeError("Erro de compilacao do Vertex Shader")


glCompileShader(fragment)
if not glGetShaderiv(fragment, GL_COMPILE_STATUS):
    error = glGetShaderInfoLog(fragment).decode()
    print(error)
    raise RuntimeError("Erro de compilacao do Fragment Shader")


# -------------------------------------------

# Attach shader objects to the program
glAttachShader(program, vertex)
glAttachShader(program, fragment)

# ------------------------------------------------------------

# Build program
glLinkProgram(program)
if not glGetProgramiv(program, GL_LINK_STATUS):
    print(glGetProgramInfoLog(program))
    raise RuntimeError('Linking error')
    
# Make program the default program
glUseProgram(program)

def load_model_from_file(filename):
    """Loads a Wavefront OBJ file. """
    objects = {}
    vertices = []
    normals = []
    texture_coords = []
    faces = []

    material = None

    # abre o arquivo obj para leitura
    for line in open(filename, "r"): ## para cada linha do arquivo .obj
        if line.startswith('#'): continue ## ignora comentarios
        values = line.split() # quebra a linha por espaço
        if not values: continue


        ### recuperando vertices
        if values[0] == 'v':
            vertices.append(values[1:4])

        ### recuperando vertices
        if values[0] == 'vn':
            normals.append(values[1:4])

        ### recuperando coordenadas de textura
        elif values[0] == 'vt':
            texture_coords.append(values[1:3])

        ### recuperando faces 
        elif values[0] in ('usemtl', 'usemat'):
            material = values[1]
        elif values[0] == 'f':
            face = []
            face_texture = []
            face_normals = []
            for v in values[1:]:
                w = v.split('/')
                face.append(int(w[0]))
                face_normals.append(int(w[2]))
                if len(w) >= 2 and len(w[1]) > 0:
                    face_texture.append(int(w[1]))
                else:
                    face_texture.append(0)

            faces.append((face, face_texture, face_normals, material))

    model = {}
    model['vertices'] = vertices
    model['texture'] = texture_coords
    model['faces'] = faces
    model['normals'] = normals

    return model



glEnable(GL_TEXTURE_2D)
qtd_texturas = 10
textures = glGenTextures(qtd_texturas)

def load_texture_from_file(texture_id, img_textura):
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    img = Image.open(img_textura)
    img_width = img.size[0]
    img_height = img.size[1]
    image_data = img.tobytes("raw", "RGB", 0, -1)
    #image_data = np.array(list(img.getdata()), np.uint8)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img_width, img_height, 0, GL_RGB, GL_UNSIGNED_BYTE, image_data)

# -------------------------------------------------------------------------------

vertices_list = []    
normals_list = []    
textures_coord_list = []

# ---------------------------------------------------------------------------

modelo = load_model_from_file('caixa\\caixa.obj')

### inserindo vertices do modelo no vetor de vertices
print('Processando modelo cube.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
    for normal_id in face[2]:
        normals_list.append( modelo['normals'][normal_id-1] )
print('Processando modelo cube.obj. Vertice final:',len(vertices_list))

### inserindo coordenadas de textura do modelo no vetor de texturas


### carregando textura equivalente e definindo um id (buffer): use um id por textura!
load_texture_from_file(0,'caixa\\caixa.jpg')


# -------------------------

modelo = load_model_from_file('luz\\luz.obj')

### inserindo vertices do modelo no vetor de vertices
print('Processando modelo luz.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
    for normal_id in face[2]:
        normals_list.append( modelo['normals'][normal_id-1] )
print('Processando modelo luz.obj. Vertice final:',len(vertices_list))

### inserindo coordenadas de textura do modelo no vetor de texturas


### carregando textura equivalente e definindo um id (buffer): use um id por textura!
load_texture_from_file(1,'luz\\luz.png')

# ----------------------------------------------------

modelo = load_model_from_file('baleia\\baleia.obj')

### inserindo vertices do modelo no vetor de vertices
print('Processando modelo baleia.obj. Vertice inicial:',len(vertices_list))
for face in modelo['faces']:
    for vertice_id in face[0]:
        vertices_list.append( modelo['vertices'][vertice_id-1] )
    for texture_id in face[1]:
        textures_coord_list.append( modelo['texture'][texture_id-1] )
    for normal_id in face[2]:
        normals_list.append( modelo['normals'][normal_id-1] )
print('Processando modelo baleia.obj. Vertice final:',len(vertices_list))

### inserindo coordenadas de textura do modelo no vetor de texturas


### carregando textura equivalente e definindo um id (buffer): use um id por textura!
load_texture_from_file(2,'baleia\\baleia.jpg')

# ----------------------------------------------------------------------------

# Request a buffer slot from GPU
buffer = glGenBuffers(3)


# ---------------------------------------------------------------------------


vertices = np.zeros(len(vertices_list), [("position", np.float32, 3)])
vertices['position'] = vertices_list


# Upload data
glBindBuffer(GL_ARRAY_BUFFER, buffer[0])
glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices, GL_STATIC_DRAW)
stride = vertices.strides[0]
offset = ctypes.c_void_p(0)
loc_vertices = glGetAttribLocation(program, "position")
glEnableVertexAttribArray(loc_vertices)
glVertexAttribPointer(loc_vertices, 3, GL_FLOAT, False, stride, offset)


# --------------------------------------------------------------------------------

textures = np.zeros(len(textures_coord_list), [("position", np.float32, 2)]) # duas coordenadas
textures['position'] = textures_coord_list


# Upload data
glBindBuffer(GL_ARRAY_BUFFER, buffer[1])
glBufferData(GL_ARRAY_BUFFER, textures.nbytes, textures, GL_STATIC_DRAW)
stride = textures.strides[0]
offset = ctypes.c_void_p(0)
loc_texture_coord = glGetAttribLocation(program, "texture_coord")
glEnableVertexAttribArray(loc_texture_coord)
glVertexAttribPointer(loc_texture_coord, 2, GL_FLOAT, False, stride, offset)




# ----------------------------------------------------------------------------------


normals = np.zeros(len(normals_list), [("position", np.float32, 3)]) # três coordenadas
normals['position'] = normals_list


# Upload coordenadas normals de cada vertice
glBindBuffer(GL_ARRAY_BUFFER, buffer[2])
glBufferData(GL_ARRAY_BUFFER, normals.nbytes, normals, GL_STATIC_DRAW)
stride = normals.strides[0]
offset = ctypes.c_void_p(0)
loc_normals_coord = glGetAttribLocation(program, "normals")
glEnableVertexAttribArray(loc_normals_coord)
glVertexAttribPointer(loc_normals_coord, 3, GL_FLOAT, False, stride, offset)



# --------------------------------------------------------------------------

ka = 0.1 # coeficiente de reflexao ambiente do modelo
kd = 0.5 # coeficiente de reflexao difusa do modelo
ks = 0.9 # coeficiente de reflexao especular do modelo

    
def desenha_luz(t_x, t_y, t_z):
    # Ângulos de rotação (ajustáveis conforme necessário)
    angle_x = 0.0
    angle_y = 0.0
    angle_z = 0.0

    # Escala
    s_x = 1
    s_y = 1
    s_z = 1

    # Cria a matriz model usando a nova função
    mat_model = model(angle_x, angle_y, angle_z, t_x, t_y, t_z, s_x, s_y, s_z)

    # Envia a matriz model para a GPU
    loc_model = glGetUniformLocation(program, "model")
    glUniformMatrix4fv(loc_model, 1, GL_FALSE, mat_model)   

    # Define os parâmetros de iluminação do modelo
    ka = 1  # Coeficiente de reflexão ambiente
    kd = 1  # Coeficiente de reflexão difusa

    # Envia ka e kd para a GPU
    loc_ka = glGetUniformLocation(program, "ka")
    glUniform1f(loc_ka, ka)

    loc_kd = glGetUniformLocation(program, "kd")
    glUniform1f(loc_kd, kd)

    if especular:
        ks = 1      # Coeficiente de reflexão especular
        ns = 1000.0 # Expoente de reflexão especular

        loc_ks = glGetUniformLocation(program, "ks")
        glUniform1f(loc_ks, ks)

        loc_ns = glGetUniformLocation(program, "ns")
        glUniform1f(loc_ns, ns)

    # Envia a posição da luz para a GPU
    loc_light_pos = glGetUniformLocation(program, "lightPos")
    glUniform3f(loc_light_pos, t_x, t_y, t_z)

    # Define a textura do modelo
    glBindTexture(GL_TEXTURE_2D, 0)

    # Desenha o modelo
    glDrawArrays(GL_TRIANGLES, 36, 36)








# -------------------------------------------------------------------


cameraPos   = glm.vec3(0.0,  0.0,  15.0);
cameraFront = glm.vec3(0.0,  0.0, -1.0);
cameraUp    = glm.vec3(0.0,  1.0,  0.0);


polygonal_mode = False
cull = False

def key_event(window,key,scancode,action,mods):
    global cameraPos, cameraFront, cameraUp, polygonal_mode, cull
    global ns_inc, ka, ks, kd
    
    cameraSpeed = 1
    if key == 87 and (action==1 or action==2): # tecla W
        cameraPos += cameraSpeed * cameraFront
    
    if key == 83 and (action==1 or action==2): # tecla S
        cameraPos -= cameraSpeed * cameraFront
    
    if key == 65 and (action==1 or action==2): # tecla A
        cameraPos -= glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed
        
    if key == 68 and (action==1 or action==2): # tecla D
        cameraPos += glm.normalize(glm.cross(cameraFront, cameraUp)) * cameraSpeed
        
    if key == 80 and action==1 and polygonal_mode==True:
        polygonal_mode=False
    else:
        if key == 80 and action==1 and polygonal_mode==False:
            polygonal_mode=True

    if key == 67 and action==1 and cull==True: # tecla C
        cull=False
    else:
        if key == 67 and action==1 and cull==False: #tecla C
            cull=True

    if key == 85 and (action==1 or action==2): # tecla U - desliga/liga iluminacao ambiente 
        if ka == 0: ka=0.3
        else: ka=0

    if key == 73 and (action==1 or action==2): # tecla I - desliga/liga reflexao difusa
        if kd == 0: kd=0.5
        else: kd=0

    if especular:
            
        if key == 79 and (action==1 or action==2): # tecla O - desliga/liga reflexao especular
            if ks == 0: ks=0.7
            else: ks=0
    
        if key == 265 and (action==1 or action==2): # tecla pra cima
            ns_inc = ns_inc * 2
            
        if key == 264 and (action==1 or action==2): # tecla pra baixo
            ns_inc = ns_inc / 2
        
firstMouse = True
yaw = -90.0 
pitch = 0.0
lastX =  largura/2
lastY =  altura/2

def mouse_event(window, xpos, ypos):
    global firstMouse, cameraFront, yaw, pitch, lastX, lastY
    if firstMouse:
        lastX = xpos
        lastY = ypos
        firstMouse = False

    xoffset = xpos - lastX
    yoffset = lastY - ypos
    lastX = xpos
    lastY = ypos

    sensitivity = 0.3 
    xoffset *= sensitivity
    yoffset *= sensitivity

    yaw += xoffset;
    pitch += yoffset;

    
    if pitch >= 90.0: pitch = 90.0
    if pitch <= -90.0: pitch = -90.0

    front = glm.vec3()
    front.x = math.cos(glm.radians(yaw)) * math.cos(glm.radians(pitch))
    front.y = math.sin(glm.radians(pitch))
    front.z = math.sin(glm.radians(yaw)) * math.cos(glm.radians(pitch))
    cameraFront = glm.normalize(front)


    
glfw.set_key_callback(window,key_event)
glfw.set_cursor_pos_callback(window, mouse_event)


# -----------------------------------------------------------------------------------------

def model(angle_x, angle_y, angle_z, t_x, t_y, t_z, s_x, s_y, s_z):
    # Funções de rotação para cada eixo
    def rotacao_x(angle):
        rad = math.radians(angle)
        return np.array([
            [1, 0, 0, 0],
            [0, math.cos(rad), -math.sin(rad), 0],
            [0, math.sin(rad), math.cos(rad), 0],
            [0, 0, 0, 1]
        ])
    
    def rotacao_y(angle):
        rad = math.radians(angle)
        return np.array([
            [math.cos(rad), 0, math.sin(rad), 0],
            [0, 1, 0, 0],
            [-math.sin(rad), 0, math.cos(rad), 0],
            [0, 0, 0, 1]
        ])
    
    def rotacao_z(angle):
        rad = math.radians(angle)
        return np.array([
            [math.cos(rad), -math.sin(rad), 0, 0],
            [math.sin(rad), math.cos(rad), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
    
    # Matriz identidade inicial
    matrix_transform = np.identity(4)
    
    # Aplicando rotações em ordem X, Y, Z
    matrix_transform = matrix_transform @ rotacao_x(angle_x)
    matrix_transform = matrix_transform @ rotacao_y(angle_y)
    matrix_transform = matrix_transform @ rotacao_z(angle_z)
    
    # Aplicando translação
    translation = np.array([
        [1, 0, 0, t_x],
        [0, 1, 0, t_y],
        [0, 0, 1, t_z],
        [0, 0, 0, 1]
    ])
    matrix_transform = matrix_transform @ translation
    
    # Aplicando escala
    scaling = np.array([
        [s_x, 0, 0, 0],
        [0, s_y, 0, 0],
        [0, 0, s_z, 0],
        [0, 0, 0, 1]
    ])
    matrix_transform = matrix_transform @ scaling
    
    return matrix_transform


def view():
    global cameraPos, cameraFront, cameraUp
    mat_view = glm.lookAt(cameraPos, cameraPos + cameraFront, cameraUp);
    mat_view = np.array(mat_view)
    return mat_view

def projection():
    global altura, largura
    # perspective parameters: fovy, aspect, near, far
    mat_projection = glm.perspective(glm.radians(45.0), largura/altura, 0.1, 1000.0)
    mat_projection = np.array(mat_projection)    
    return mat_projection

# -------------------------------------------------------------------------------------------

glfw.show_window(window)
glfw.set_cursor_pos(window, lastX, lastY)

# ------------------------------------------------------------


obj_caixa = Objeto()
obj_caixa.carrega_obj("caixa\\caixa.obj")
obj_caixa.carrega_textura("caixa\\caixa.jpg")


glEnable(GL_DEPTH_TEST) ### importante para 3D
glLightModelfv ( GL_LIGHT_MODEL_TWO_SIDE , GL_FALSE ) 

   
ang = 0.1
ns_inc = 32
    
while not glfw.window_should_close(window):

    glfw.poll_events() 
    
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    glClearColor(0.2, 0.2, 0.2, 1.0)
    
    if polygonal_mode==True:
        glPolygonMode(GL_FRONT_AND_BACK,GL_LINE)
    if polygonal_mode==False:
        glPolygonMode(GL_FRONT_AND_BACK,GL_FILL)


    if cull == True:
        glEnable(GL_CULL_FACE) 
        # glCullFace(GL_FRONT)
        glCullFace(GL_BACK)
    else:
        glDisable(GL_CULL_FACE) 
    

    # desenha_caixa()   

    # desenha_baleia()

    obj_caixa.desenha(program, model(0, 0, 0, 
                                     -1.5, 0, 0, 
                                     1, 1, 1))
    
    obj_caixa.desenha(program, model(0, 0, 0, 
                                     1.5, 0, 0, 
                                     1, 1, 1))
    # obj_caixa.desenha(program, )
    

    if especular: 
        ang += 0.01


    desenha_luz(math.cos(ang), math.sin(ang), 3.0)   


    
    mat_view = view()
    loc_view = glGetUniformLocation(program, "view")
    glUniformMatrix4fv(loc_view, 1, GL_TRUE, mat_view)

    mat_projection = projection()
    loc_projection = glGetUniformLocation(program, "projection")
    glUniformMatrix4fv(loc_projection, 1, GL_TRUE, mat_projection)    

    if especular:
        # atualizando a posicao da camera/observador na GPU para calculo da reflexao especular
        loc_view_pos = glGetUniformLocation(program, "viewPos") # recuperando localizacao da variavel viewPos na GPU
        glUniform3f(loc_view_pos, cameraPos[0], cameraPos[1], cameraPos[2]) ### posicao da camera/observador (x,y,z)
    
    glfw.swap_buffers(window)

glfw.terminate()