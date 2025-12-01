import unittest
import os
import shutil
import sys
import textwrap
import pandas as pd

# Ajustar path para importar src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from extract_features import analyze_file

class TestFeatureExtractorPro(unittest.TestCase):
    
    TEMP_DIR = "test_samples_pro"

    @classmethod
    def setUpClass(cls):
        """Prepara el entorno de pruebas."""
        if os.path.exists(cls.TEMP_DIR):
            shutil.rmtree(cls.TEMP_DIR)
        os.makedirs(cls.TEMP_DIR)

    @classmethod
    def tearDownClass(cls):
        """Limpia todo al terminar."""
        if os.path.exists(cls.TEMP_DIR):
            shutil.rmtree(cls.TEMP_DIR)

    def create_file(self, filename, content):
        """Helper para crear archivos limpiando indentación."""
        path = os.path.join(self.TEMP_DIR, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(textwrap.dedent(content).strip())
        return path

    # ==========================================
    # 1. PYTHON TESTS (3 Casos)
    # ==========================================
    
    def test_py_01_os_system(self):
        """Caso 1 Python: Inyección de Comandos clásica."""
        content = """
        import os
        def delete_logs(user_input):
            # Vulnerable a inyección
            os.system("rm -rf " + user_input)
        """
        res = analyze_file(self.create_file("py_vuln_1.py", content), "py_vuln_1.py")
        self.assertEqual(res['uses_dangerous_funcs'], 1, "Debe detectar os.system")
        self.assertEqual(res['nesting_depth'], 1)

    def test_py_02_eval_exec(self):
        """Caso 2 Python: Ejecución dinámica de código."""
        content = """
        def calc(expression):
            # Vulnerable a inyección de código
            return eval(expression)
            
        def dynamic_load(code):
            exec(code)
        """
        res = analyze_file(self.create_file("py_vuln_2.py", content), "py_vuln_2.py")
        self.assertEqual(res['uses_dangerous_funcs'], 2, "Debe detectar eval y exec")

    def test_py_03_safe_complex(self):
        """Caso 3 Python: Código seguro pero con anidamiento profundo."""
        content = """
        def procesar_datos(lista):
            if lista:
                for item in lista:
                    try:
                        if item > 0:
                            print(item)
                    except:
                        pass
        """
        res = analyze_file(self.create_file("py_safe.py", content), "py_safe.py")
        self.assertEqual(res['uses_dangerous_funcs'], 0, "No debe haber vulnerabilidades")
        
        # CORRECCIÓN: Esperamos 5.
        # Explicación: def(0) -> if(1) -> for(2) -> try(3) -> if(4) -> print(5)
        # La línea más profunda ('print') está indentada 5 niveles.
        self.assertEqual(res['nesting_depth'], 5)

    # ==========================================
    # 2. JAVA TESTS (3 Casos)
    # ==========================================

    def test_java_01_runtime(self):
        """Caso 1 Java: Runtime.exec()."""
        content = """
        class Danger {
            public void runCommand(String cmd) {
                try {
                    Runtime.getRuntime().exec(cmd);
                } catch(Exception e) {}
            }
        }
        """
        res = analyze_file(self.create_file("J_Vuln1.java", content), "J_Vuln1.java")
        self.assertEqual(res['uses_dangerous_funcs'], 1, "Debe detectar .exec(")

    def test_java_02_processbuilder(self):
        """Caso 2 Java: ProcessBuilder (El que fallaba antes)."""
        content = """
        public class VulnClass {
            public static void main(String[] args) {
                // Comentario mencionando ProcessBuilder que NO debe contar
                ProcessBuilder pb = new ProcessBuilder("cmd", "/c", "dir");
                pb.start();
            }
        }
        """
        res = analyze_file(self.create_file("J_Vuln2.java", content), "J_Vuln2.java")
        # Ahora el regex busca 'new ProcessBuilder', así que debe ser 1 exacto
        self.assertEqual(res['uses_dangerous_funcs'], 1, "Debe detectar solo la instanciación de ProcessBuilder")

    def test_java_03_safe_deep(self):
        """Caso 3 Java: Código seguro anidado."""
        content = """
        public class Safe {
            void loop() {            
                if (true) {          
                    while (true) {   
                         System.out.println("Safe");
                    }
                }
            }
        }
        """
        res = analyze_file(self.create_file("J_Safe.java", content), "J_Safe.java")
        self.assertEqual(res['uses_dangerous_funcs'], 0)
        
        # CORRECCIÓN: Esperamos 4.
        # Explicación: Class{ (1) -> Method{ (2) -> If{ (3) -> While{ (4)
        self.assertEqual(res['nesting_depth'], 4)

    # ==========================================
    # 3. PHP TESTS (3 Casos)
    # ==========================================

    def test_php_01_shell_exec(self):
        """Caso 1 PHP: shell_exec."""
        content = """
        <?php
        $output = shell_exec('ls -l');
        ?>
        """
        res = analyze_file(self.create_file("vuln.php", content), "vuln.php")
        self.assertEqual(res['uses_dangerous_funcs'], 1)

    def test_php_02_system_false_positive(self):
        """Caso 2 PHP: 'system' vs 'filesystem' (Prueba de Word Boundary)."""
        content = """
        <?php
        system("rm -rf /");       // Peligroso (1)
        $var = filesystem_check(); // Seguro (0) - No debe confundirse
        ?>
        """
        res = analyze_file(self.create_file("boundary.php", content), "boundary.php")
        self.assertEqual(res['uses_dangerous_funcs'], 1, "Solo debe detectar 'system', no 'filesystem'")

    def test_php_03_safe(self):
        """Caso 3 PHP: HTML y Echo."""
        content = """
        <?php echo "Hola Mundo"; ?>
        """
        res = analyze_file(self.create_file("safe.php", content), "safe.php")
        self.assertEqual(res['uses_dangerous_funcs'], 0)

    # ==========================================
    # 4. JAVASCRIPT TESTS (3 Casos)
    # ==========================================

    def test_js_01_eval(self):
        """Caso 1 JS: Eval directo."""
        content = "eval('2 + 2');"
        res = analyze_file(self.create_file("vuln1.js", content), "vuln1.js")
        self.assertEqual(res['uses_dangerous_funcs'], 1)

    def test_js_02_settimeout_string(self):
        """Caso 2 JS: setTimeout con string (implica eval)."""
        content = """
        setTimeout("alert('Hacked')", 1000);
        """
        res = analyze_file(self.create_file("vuln2.js", content), "vuln2.js")
        self.assertEqual(res['uses_dangerous_funcs'], 1)

    def test_js_03_safe_nesting(self):
        """Caso 3 JS: Callbacks anidados (Callback Hell)."""
        content = """
        function a() {
            if (x) {
                items.forEach(i => {
                    console.log(i);
                });
            }
        }
        """
        res = analyze_file(self.create_file("safe.js", content), "safe.js")
        # Function(1) -> If(2) -> forEach(3)
        self.assertTrue(res['nesting_depth'] >= 3)
        self.assertEqual(res['uses_dangerous_funcs'], 0)

    # ==========================================
    # 5. C++ TESTS (3 Casos)
    # ==========================================

    def test_cpp_01_system(self):
        """Caso 1 C++: system()."""
        content = """
        #include <stdlib.h>
        int main() {
            system("pause");
            return 0;
        }
        """
        res = analyze_file(self.create_file("vuln1.cpp", content), "vuln1.cpp")
        self.assertEqual(res['uses_dangerous_funcs'], 1)

    def test_cpp_02_strcpy(self):
        """Caso 2 C++: Buffer Overflow clásico (strcpy)."""
        content = """
        #include <string.h>
        void copy(char* src) {
            char dest[10];
            strcpy(dest, src); // Peligroso
        }
        """
        res = analyze_file(self.create_file("vuln2.cpp", content), "vuln2.cpp")
        self.assertEqual(res['uses_dangerous_funcs'], 1)

    def test_cpp_03_safe(self):
        """Caso 3 C++: Hello World."""
        content = """
        #include <iostream>
        int main() {
            std::cout << "Hello World";
            return 0;
        }
        """
        res = analyze_file(self.create_file("safe.cpp", content), "safe.cpp")
        self.assertEqual(res['uses_dangerous_funcs'], 0)

if __name__ == '__main__':
    unittest.main(verbosity=2)