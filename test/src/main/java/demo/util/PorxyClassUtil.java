package demo.util;

import lombok.val;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.lang.classfile.ClassFile;
import java.lang.classfile.MethodModel;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;
import java.lang.invoke.MethodHandles;
import java.lang.invoke.MethodType;
import java.lang.reflect.AccessFlag;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/25
 */
@SuppressWarnings({"preview"})
public class PorxyClassUtil extends ClassLoader {
    private static final ClassFile cf = ClassFile.of();
    private static final MethodHandles.Lookup lookup = MethodHandles.lookup();
    private final boolean saveFile;

    public PorxyClassUtil(boolean saveFile) {
        super(PorxyClassUtil.class.getClassLoader());
        this.saveFile = saveFile;
    }

    private static byte[] proxy0(Class<?> clazz, String simpleNameProxy, boolean saveFile) throws IOException {
        val nameWithPackage = clazz.getName().replace(".", "/");
        val index = nameWithPackage.lastIndexOf('/');
        val className = nameWithPackage.substring(index + 1);
        val path = Path.of(clazz.getResource(className + ".class").getPath());
        val classModel = cf.parse(path);
        val build = cf.build(ClassDesc.ofInternalName(simpleNameProxy), classBuilder -> {
            classBuilder.withSuperclass(ClassFileUtil.ofClass(clazz));
            for (MethodModel methodModel : classModel.methods()) {
                val methodName = methodModel.methodName().stringValue();
                val flags = methodModel.flags();
                if (flags.has(AccessFlag.PRIVATE) || flags.has(AccessFlag.STATIC) || flags.has(AccessFlag.FINAL)) {
                    continue;
                }
                val methodTypeDesc = methodModel.methodTypeSymbol();
                classBuilder.withMethod(methodName, methodTypeDesc, flags.flagsMask(), methodBuilder -> {
                    methodBuilder.withCode(codeBuilder -> {
                        val returnType = methodTypeDesc.returnType();
                        val returnKind = TypeKind.fromDescriptor(returnType.descriptorString());
                        ClassFileUtil.println(codeBuilder, className + "#" + methodName + " start");
                        ClassFileUtil.callSuper(codeBuilder, clazz, methodName, methodTypeDesc);
                        if (returnKind == TypeKind.VOID) {
                            ClassFileUtil.println(codeBuilder, className + "#" + methodName + " end");
                        } else {
                            val count = methodTypeDesc.parameterCount() + 1;
                            codeBuilder.storeLocal(returnKind, count);
                            ClassFileUtil.print(codeBuilder, className + "#" + methodName + " end, return: ");
                            ClassFileUtil.printSlot(codeBuilder, returnType, count);
                            codeBuilder.loadLocal(returnKind, count);
                        }
                        codeBuilder.return_(returnKind);
                    });
                });
            }
        });
        if (saveFile) {
            val packageName = nameWithPackage.substring(0, index);
            val classDir = new File("./test/target/classes/" + packageName);
            if (!classDir.exists()) {
                classDir.mkdirs();
            }
            val classNameProxy = className + "__Proxy" + ".class";
            val classPath = new File(classDir, classNameProxy);
            Files.write(classPath.toPath(), build);
        }
        return build;
    }

    @SuppressWarnings("unchecked")
    public static <T> T proxyLookup(Class<T> clazz) throws IOException {
        val simpleNameProxy = "demo/util/" + clazz.getName().replace(".", "$");
        val build = proxy0(clazz, simpleNameProxy, true);
        try {
            val aClass = (Class<T>) lookup.defineClass(build);
            val constructor = lookup.findConstructor(aClass, MethodType.methodType(void.class));
            return (T) constructor.invoke();
        } catch (Throwable e) {
            throw new RuntimeException(e);
        }
    }

    @SuppressWarnings("unchecked")
    public <T> T proxy(Class<T> clazz) throws Throwable {
        val nameWithPackage = clazz.getName().replace(".", "/");
        val simpleNameProxy = nameWithPackage + "__Proxy";
        val build = proxy0(clazz, simpleNameProxy, saveFile);
        try {
            val aClass = (Class<T>) defineClass(
                    clazz.getName() + "__Proxy",
                    build
            );
            val constructor = aClass.getConstructor();
            return constructor.newInstance();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public Class<?> defineClass(String name, byte[] b) {
        return defineClass(name, b, 0, b.length);
    }

    public void helloworld() throws IOException {
        val build = cf.build(ClassDesc.of(
                // "demo",
                "ClassFileApiDynamicTest"
        ), clazz -> {
            clazz.withMethod("main", MethodTypeDesc.of(
                    ClassDesc.ofDescriptor(void.class.descriptorString()),
                    ClassDesc.ofDescriptor(String[].class.descriptorString())
            ), ClassFile.ACC_PUBLIC | ClassFile.ACC_STATIC, method -> {
                method.withCode(code -> {
                    code.getstatic(
                            ClassDesc.ofDescriptor(System.class.descriptorString()),
                            "out",
                            ClassDesc.ofDescriptor(PrintStream.class.descriptorString())
                    );
                    code.ldc("Hello, World!");
                    code.invokevirtual(
                            ClassDesc.ofDescriptor(PrintStream.class.descriptorString()),
                            "println",
                            MethodTypeDesc.of(
                                    ClassDesc.ofDescriptor(void.class.descriptorString()),
                                    ClassDesc.ofDescriptor(String.class.descriptorString())
                            )
                    );
                    code.return_();
                });
            });
        });
        Files.write(Path.of("./test/target/classes/demo/ClassFileApiDynamicTest.class"), build);
    }

    public void helloworld2() throws IOException {
        val build = cf.build(ClassDesc.of(
                // "demo",
                "ClassFileApiDynamicTest"
        ), clazz -> {
            clazz.withMethod("main", ClassFileUtil.ofMethod(
                    void.class, String[].class
            ), ClassFile.ACC_PUBLIC | ClassFile.ACC_STATIC, method -> {
                method.withCode(code -> {
                    ClassFileUtil.println(code, "Hello, World!");
                    code.return_();
                });
            });
        });
        Files.write(Path.of("./test/target/classes/demo/ClassFileApiDynamicTest.class"), build);
    }
}
