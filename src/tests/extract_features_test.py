import unittest
import os
import shutil
import sys
import ast
import textwrap  # <--- IMPORTANTE: Necesario para limpiar la indentación

# Añadimos el directorio padre al path para poder importar src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.extract_features import analyze_single_file, SecurityVisitor

class TestSecurityVisitor(unittest.TestCase):
    """
    Tests Unitarios para la clase SecurityVisitor (AST).
    Verifica la lógica de detección de anidamiento y funciones peligrosas.
    """

    def _visit_code(self, code_str):
        """Helper para parsear código y ejecutar el visitante."""
        # Limpiamos la indentación antes de parsear
        clean_code = textwrap.dedent(code_str).strip()
        tree = ast.parse(clean_code)
        visitor = SecurityVisitor()
        visitor.visit(tree)
        return visitor

    def test_nesting_ignores_function_def(self):
        """Verifica que 'def' NO aumente el nivel de anidamiento."""
        code = """
        def mi_funcion():
            if True:        # Nivel 1
                print("x")
        """
        visitor = self._visit_code(code)
        # def (0) -> if (1). Max depth debe ser 1.
        self.assertEqual(visitor.max_depth, 1, "El 'def' no debería contar como anidamiento.")

    def test_nesting_deep_logic(self):
        """Verifica anidamiento profundo real."""
        code = """
        if A:                   # 1
            for i in range(10): # 2
                try:            # 3
                    pass
                except:
                    pass
        """
        visitor = self._visit_code(code)
        self.assertEqual(visitor.max_depth, 3)

    def test_detect_eval(self):
        """Debe detectar la función 'eval'."""
        code = "x = eval('2 + 2')"
        visitor = self._visit_code(code)
        self.assertEqual(visitor.dangerous_count, 1)

    def test_detect_exec(self):
        """Debe detectar la función 'exec'."""
        code = "exec('import os')"
        visitor = self._visit_code(code)
        self.assertEqual(visitor.dangerous_count, 1)

    def test_detect_os_system(self):
        """Debe detectar 'system' (comunmente os.system)."""
        code = """
        import os
        os.system("rm -rf /")
        """
        visitor = self._visit_code(code)
        self.assertEqual(visitor.dangerous_count, 1)

    def test_safe_code_zero_danger(self):
        """Código normal no debe disparar alertas."""
        code = "print('Hola mundo')"
        visitor = self._visit_code(code)
        self.assertEqual(visitor.dangerous_count, 0)


class TestIntegrationAnalysis(unittest.TestCase):
    """
    Tests de Integración: Crea archivos reales y prueba 'analyze_single_file'.
    Verifica que Lizard y AST trabajen juntos y generen el diccionario correcto.
    """
    
    TEMP_DIR = "test_samples_temp"

    @classmethod
    def setUpClass(cls):
        """Crea un directorio temporal para las pruebas."""
        if os.path.exists(cls.TEMP_DIR):
            shutil.rmtree(cls.TEMP_DIR)
        os.makedirs(cls.TEMP_DIR)

    @classmethod
    def tearDownClass(cls):
        """Borra el directorio temporal al finalizar."""
        if os.path.exists(cls.TEMP_DIR):
            shutil.rmtree(cls.TEMP_DIR)

    def create_dummy_file(self, filename, content):
        path = os.path.join(self.TEMP_DIR, filename)
        # Limpiamos la indentación antes de escribir al archivo
        clean_content = textwrap.dedent(content).strip()
        with open(path, "w", encoding='utf-8') as f:
            f.write(clean_content)
        return path

    def test_analyze_vulnerable_file(self):
        """Prueba un archivo que debería ser marcado como VULNERABLE (1)."""
        filename = "CWE78_OS_Command_Injection__bad.py"
        content = """
        import os
        def bad_code():
            os.system("ls") # Danger!
        """
        path = self.create_dummy_file(filename, content)
        result = analyze_single_file(path, filename)
        
        self.assertIsNotNone(result)
        # Verificamos etiqueta por nombre
        self.assertEqual(result['is_vulnerable'], 1, "Debería detectar 'bad' en el nombre.")
        # Verificamos detección de AST (Si falla aquí, es que el AST no parseó bien el file)
        self.assertEqual(result['uses_dangerous_funcs'], 1, "Debería detectar os.system")
        # Verificamos Lizard
        self.assertEqual(result['num_functions'], 1)

    def test_analyze_safe_file(self):
        """Prueba un archivo que debería ser marcado como SEGURO (0)."""
        filename = "CWE78_OS_Command_Injection__good.py"
        content = """
        def good_code():
            print("Safe")
        """
        path = self.create_dummy_file(filename, content)
        result = analyze_single_file(path, filename)
        
        self.assertEqual(result['is_vulnerable'], 0, "Debería detectar 'good' en el nombre.")
        self.assertEqual(result['uses_dangerous_funcs'], 0)

    def test_syntax_error_handling(self):
        """Prueba que el script no explote con errores de sintaxis."""
        filename = "broken.py"
        content = "def funcion_rota( print('falta parentesis'"
        path = self.create_dummy_file(filename, content)
        
        # No debería lanzar excepción, sino retornar métricas parciales o seguras
        result = analyze_single_file(path, filename)
        
        self.assertIsNotNone(result)
        self.assertIn('loc', result)

if __name__ == '__main__':
    unittest.main(verbosity=2)