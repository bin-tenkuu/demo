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
import java.lang.reflect.AccessFlag;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/25
 */
@SuppressWarnings({"preview", "StringTemplateMigration"})
public class PorxyClassUtil extends ClassLoader {
    private final ClassFile cf = ClassFile.of();
    private final boolean saveFile;

    public PorxyClassUtil(boolean saveFile) {
        super(PorxyClassUtil.class.getClassLoader());
        this.saveFile = saveFile;
    }

    @SuppressWarnings("unchecked")
    public <T> T proxy(Class<T> clazz) throws IOException {
        val nameWithPackage = clazz.getName().replace(".", "/");
        val index = nameWithPackage.lastIndexOf('/');
        val className = nameWithPackage.substring(index + 1);
        val path = Path.of(clazz.getResource(className + ".class").getPath());
        val classModel = cf.parse(path);
        val simpleNameProxy = nameWithPackage + "__Proxy";
        val build = cf.build(ClassDesc.ofInternalName(simpleNameProxy), classBuilder -> {
            classBuilder.withSuperclass(ClassFileUtil.of(clazz));
            for (MethodModel methodModel : classModel.methods()) {
                val methodName = methodModel.methodName().stringValue();
                val flags = methodModel.flags();
                if (flags.has(AccessFlag.PRIVATE) || flags.has(AccessFlag.STATIC) || flags.has(AccessFlag.FINAL)) {
                    continue;
                }
                val methodTypeDesc = methodModel.methodTypeSymbol();
                classBuilder.withMethod(
                        methodName,
                        methodTypeDesc,
                        flags.flagsMask(),
                        methodBuilder -> {
                            methodBuilder.withCode(codeBuilder -> {
                                val returnType = methodTypeDesc.returnType();
                                val returnKind = TypeKind.fromDescriptor(
                                        returnType.descriptorString());
                                ClassFileUtil.println(codeBuilder, className + "#" + methodName + " start");
                                ClassFileUtil.callSuper(codeBuilder, clazz, methodName, methodTypeDesc);
                                if (returnKind == TypeKind.VoidType) {
                                    ClassFileUtil.println(codeBuilder, className + "#" + methodName + " end");
                                    codeBuilder.return_();
                                } else {
                                    val L0 = codeBuilder.startLabel();
                                    codeBuilder.storeLocal(returnKind, 2);
                                    ClassFileUtil.getstatic(codeBuilder, System.class, "out", PrintStream.class);
                                    ClassFileUtil.print(codeBuilder,
                                            className + "#" + methodName + " end, return: ");
                                    ClassFileUtil.getstatic(codeBuilder, System.class, "out", PrintStream.class);
                                    codeBuilder.loadLocal(returnKind, 2);
                                    ClassFileUtil.invokevirtual(codeBuilder, PrintStream.class, "println",
                                            MethodTypeDesc.of(ClassFileUtil.of(void.class), returnType));
                                    codeBuilder.loadLocal(returnKind, 2);
                                    codeBuilder.return_(returnKind);
                                    val L1 = codeBuilder.endLabel();
                                    ClassFileUtil.paramVariable(codeBuilder, L0, L1, clazz, methodTypeDesc);
                                    val count = methodTypeDesc.parameterCount() + 1;
                                    codeBuilder.localVariable(count, "var" + count, returnType, L0, L1);
                                }
                            });
                        }
                );
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
        try {
            val aClass = (Class<T>) defineClass(
                    simpleNameProxy.replace("/", "."),
                    build
            );
            // MethodHandles.lookup().in(clazz).defineClass(build);
            val constructor = aClass.getConstructor();
            return constructor.newInstance();
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    public Class<?> defineClass(String name, byte[] b) {
        return defineClass(name, b, 0, b.length);
    }
}
