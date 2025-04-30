#ifndef SHADER_H
#define SHADER_H

#include <GL/glew.h>
#include <glm/glm.hpp> 
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class Shader {
private:
    using string = std::string;

    unsigned int _ID;

    void checkCompileErrors(unsigned int shader, string type) {
        int success;
        char infoLog[1024];
        if (type != "PROGRAM") {
            glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
            if(!success) {
                glGetShaderInfoLog(shader, 1024, NULL, infoLog);
                std::cout << "ERROR::SHADER_COMPILATION_ERROR of type: " << type << '\n' << infoLog << '\n';
            }
        } else {
            glGetProgramiv(shader, GL_LINK_STATUS, &success);
            if(!success) {
                glGetProgramInfoLog(shader, 1024, NULL, infoLog);
                std::cout << "ERROR::PROGRAM_LINKING_ERROR of type: " << type << '\n' << infoLog << '\n';
            }
        }
    }

public:
    Shader(const char* _vertPath, const char* _fragPath) {
        string _vertCode;
        string _fragCode;
        std::ifstream _vertShaderFile;
        std::ifstream _fragShaderFile;

        _vertShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);
        _fragShaderFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

        try {
            _vertShaderFile.open(_vertPath);
            _fragShaderFile.open(_fragPath);

            std::stringstream _vertShaderStream, _fragShaderStream;
            _vertShaderStream << _vertShaderFile.rdbuf();
            _fragShaderStream << _fragShaderFile.rdbuf();

            _vertShaderFile.close();
            _fragShaderFile.close();

            _vertCode = _vertShaderStream.str();
            _fragCode = _fragShaderStream.str();
        } catch (std::ifstream::failure e) {
            std::cout << "ERROR::SHADER::FILE_NOT_SUCCESFULLY_READ\n";
        }

        const char* _vertShaderCode = _vertCode.c_str();
        const char* _fragShaderCode = _fragCode.c_str();

        unsigned int vertex, fragment;

        vertex = glCreateShader(GL_VERTEX_SHADER);
        glShaderSource(vertex, 1, &_vertShaderCode, NULL);
        glCompileShader(vertex);
		checkCompileErrors(vertex, "VERTEX");

		fragment = glCreateShader(GL_FRAGMENT_SHADER);
		glShaderSource(fragment, 1, &_fragShaderCode, NULL);
		glCompileShader(fragment);
		checkCompileErrors(fragment, "FRAGMENT");

		_ID = glCreateProgram();
		glAttachShader(_ID, vertex);
		glAttachShader(_ID, fragment);
		glLinkProgram(_ID);
		checkCompileErrors(_ID, "PROGRAM");

		glDeleteShader(vertex);
		glDeleteShader(fragment);
    }

    ~Shader() {
		glDeleteProgram(_ID);
	}

	unsigned int get_id() const {
		return _ID;
	}

	void use() {
		glUseProgram(_ID);
	}

	void set_bool(const string& _Name, bool _Val) const {
		glUniform1i(glGetUniformLocation(_ID, _Name.c_str()), (int)_Val);
	}

	void set_int(const string& _Name, int _Val) const {
		glUniform1i(glGetUniformLocation(_ID, _Name.c_str()), _Val);
	}

	void set_float(const string& _Name, float _Val) const {
		glUniform1f(glGetUniformLocation(_ID, _Name.c_str()), _Val);
	}

	void set_vec2(const string& _Name, const glm::vec2& _Val) const {
		glUniform2fv(glGetUniformLocation(_ID, _Name.c_str()), 1, &_Val[0]);
	}

	void set_vec2(const string& _Name, float _x, float _y) const {
		glUniform2f(glGetUniformLocation(_ID, _Name.c_str()), _x, _y);
	}

	void set_vec3(const string& _Name, const glm::vec3& _Val) const {
		glUniform3fv(glGetUniformLocation(_ID, _Name.c_str()), 1, &_Val[0]);
	}

	void set_vec3(const string& _Name, float _x, float _y, float _z) const {
		glUniform3f(glGetUniformLocation(_ID, _Name.c_str()), _x, _y, _z);
	}

	void set_vec4(const string& _Name, const glm::vec4& _Val) const {
		glUniform4fv(glGetUniformLocation(_ID, _Name.c_str()), 1, &_Val[0]);
	}

	void set_vec4(const string& _Name, float _x, float _y, float _z, float _w) const {
		glUniform4f(glGetUniformLocation(_ID, _Name.c_str()), _x, _y, _z, _w);
	}

	void set_mat2(const string& _Name, const glm::mat2& _Val) const {
		glUniformMatrix2fv(glGetUniformLocation(_ID, _Name.c_str()), 1, GL_FALSE, &_Val[0][0]);
	}

	void set_mat3(const string& _Name, const glm::mat3& _Val) const {
		glUniformMatrix3fv(glGetUniformLocation(_ID, _Name.c_str()), 1, GL_FALSE, &_Val[0][0]);
	}

	void set_mat4(const string& _Name, const glm::mat4& _Val) const {
		glUniformMatrix4fv(glGetUniformLocation(_ID, _Name.c_str()), 1, GL_FALSE, &_Val[0][0]);
	}
};

#endif