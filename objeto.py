from OpenGL.GL import *
from PIL import Image
import glm
import numpy as np

class Objeto:
    def __init__(self):
        self.vertices = []
        self.textures = []
        self.normals = []
        self.faces = []
        self.texture_id = None
        self.transformation_matrix = glm.mat4(1.0)
        self.vertex_buffer = None
        self.texture_buffer = None
        self.normal_buffer = None

    def carrega_textura(self, endereco_textura):
        # Carregar textura e enviar para a GPU
        self.texture_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, self.texture_id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        img = Image.open(endereco_textura)
        img_data = np.array(img.convert("RGB"), np.uint8)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, img.width, img.height, 0, GL_RGB, GL_UNSIGNED_BYTE, img_data)

    def carrega_obj(self, endereco_obj):
        # Carregar modelo OBJ e preparar buffers
        self.vertices = []
        self.textures = []
        self.normals = []
        with open(endereco_obj, "r") as file:
            for line in file:
                if line.startswith("v "):
                    self.vertices.append(list(map(float, line.strip().split()[1:4])))
                elif line.startswith("vt "):
                    self.textures.append(list(map(float, line.strip().split()[1:3])))
                elif line.startswith("vn "):
                    self.normals.append(list(map(float, line.strip().split()[1:4])))
                elif line.startswith("f "):
                    face = line.strip().split()[1:]
                    for vertex in face:
                        v, t, n = (int(x) if x else 0 for x in vertex.split("/"))
                        self.faces.append((v, t, n))

        # Preparar buffers
        self.vertex_buffer = self._prepare_buffer(self.vertices, 3)
        self.texture_buffer = self._prepare_buffer(self.textures, 2)
        self.normal_buffer = self._prepare_buffer(self.normals, 3)

    def _prepare_buffer(self, data, size):
        buffer_data = np.array(data, dtype=np.float32)
        buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, buffer)
        glBufferData(GL_ARRAY_BUFFER, buffer_data.nbytes, buffer_data, GL_STATIC_DRAW)
        return buffer

    def desenha(self, program, model_matrix):
        # Atualizar matriz de transformação
        loc_model = glGetUniformLocation(program, "model")
        glUniformMatrix4fv(loc_model, 1, GL_FALSE, np.array(model_matrix).T)

        # Bind da textura
        if self.texture_id:
            glBindTexture(GL_TEXTURE_2D, self.texture_id)

        # Bind dos buffers
        self._bind_attribute(program, "position", self.vertex_buffer, 3)
        self._bind_attribute(program, "texture_coord", self.texture_buffer, 2)
        self._bind_attribute(program, "normals", self.normal_buffer, 3)

        # Desenhar o objeto
        print("desenha:"+str(len(self.faces)))
        glDrawArrays(GL_TRIANGLES, 0, len(self.faces))

    def _bind_attribute(self, program, attribute, buffer, size):
        loc = glGetAttribLocation(program, attribute)
        glEnableVertexAttribArray(loc)
        glBindBuffer(GL_ARRAY_BUFFER, buffer)
        glVertexAttribPointer(loc, size, GL_FLOAT, GL_FALSE, 0, None)
