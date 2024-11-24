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
        # self.indice_inicial = 0

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
        # Inicializar listas para vértices, texturas, normais e faces
        vertices_list = []
        textures_coord_list = []
        normals_list = []
        faces = []

        print(f"Lendo arquivo {endereco_obj}...")

        with open(endereco_obj, "r") as file:
            print(f"Arquivo ({endereco_obj}) encontrado.")
            for line in file:
                if line.startswith("#"):  # Ignorar comentários
                    continue
                values = line.strip().split()
                if not values:
                    continue

                if values[0] == "v":  # Vértices
                    vertices_list.append(list(map(float, values[1:4])))
                elif values[0] == "vt":  # Coordenadas de textura
                    textures_coord_list.append(list(map(float, values[1:3])))
                elif values[0] == "vn":  # Normais
                    normals_list.append(list(map(float, values[1:4])))
                elif values[0] == "f":  # Faces
                    # print(f"Face encontrada: {values[1:]}")
                    face_vertices = []
                    face_textures = []
                    face_normals = []
                    for v in values[1:]:
                        w = v.split("/")
                        face_vertices.append(int(w[0]) - 1)  # Índices começam em 1
                        if len(w) > 1 and w[1]:
                            face_textures.append(int(w[1]) - 1)
                        else:
                            face_textures.append(-1)  # Sem coordenada de textura
                        if len(w) > 2 and w[2]:
                            face_normals.append(int(w[2]) - 1)
                        else:
                            face_normals.append(-1)  # Sem normal
                    faces.append((face_vertices, face_textures, face_normals))

        if not faces:
            print("Nenhuma face foi encontrada no arquivo OBJ!")
            return

        print(f"Processando modelo {endereco_obj}...")

        # Dados finais para envio aos buffers
        final_vertices = []
        final_textures = []
        final_normals = []

        self.faces = faces
        for face in faces:
            face_vertices = [vertices_list[i] for i in face[0]]
            face_textures = [
                textures_coord_list[i] if i >= 0 and i < len(textures_coord_list) else [0.0, 0.0]
                for i in face[1]
            ]
            face_normals = [
                normals_list[i] if i >= 0 and i < len(normals_list) else [0.0, 0.0, 0.0]
                for i in face[2]
            ]

            if len(face[0]) == 3:  # Triângulo
                final_vertices.extend(face_vertices)
                final_textures.extend(face_textures)
                final_normals.extend(face_normals)
            elif len(face[0]) == 4:  # Quadrilátero (dividir em dois triângulos)
                final_vertices.extend([face_vertices[0], face_vertices[1], face_vertices[2],
                                    face_vertices[0], face_vertices[2], face_vertices[3]])
                final_textures.extend([face_textures[0], face_textures[1], face_textures[2],
                                    face_textures[0], face_textures[2], face_textures[3]])
                final_normals.extend([face_normals[0], face_normals[1], face_normals[2],
                                    face_normals[0], face_normals[2], face_normals[3]])

        print(f"Total de vertices processados: {len(final_vertices)}")

        # Enviar dados para buffers da GPU
        self.vertex_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.vertex_buffer)
        glBufferData(GL_ARRAY_BUFFER, np.array(final_vertices, dtype=np.float32).nbytes,
                    np.array(final_vertices, dtype=np.float32), GL_STATIC_DRAW)

        self.texture_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.texture_buffer)
        glBufferData(GL_ARRAY_BUFFER, np.array(final_textures, dtype=np.float32).nbytes,
                    np.array(final_textures, dtype=np.float32), GL_STATIC_DRAW)

        self.normal_buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, self.normal_buffer)
        glBufferData(GL_ARRAY_BUFFER, np.array(final_normals, dtype=np.float32).nbytes,
                    np.array(final_normals, dtype=np.float32), GL_STATIC_DRAW)


    def _prepare_buffer(self, data, size):
        buffer_data = np.array(data, dtype=np.float32)
        buffer = glGenBuffers(1)
        glBindBuffer(GL_ARRAY_BUFFER, buffer)
        glBufferData(GL_ARRAY_BUFFER, buffer_data.nbytes, buffer_data, GL_STATIC_DRAW)
        return buffer

    def desenha(self, program, model_matrix = glm.mat4(1.0)):
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
        # print("qtd faces: "+str(len(self.faces)))
        glDrawArrays(GL_TRIANGLES, 0, 3*len(self.faces))

    def _bind_attribute(self, program, attribute, buffer, size):
        loc = glGetAttribLocation(program, attribute)
        glEnableVertexAttribArray(loc)
        glBindBuffer(GL_ARRAY_BUFFER, buffer)
        glVertexAttribPointer(loc, size, GL_FLOAT, GL_FALSE, 0, None)
