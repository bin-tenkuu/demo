package demo.util;

import lombok.val;

import java.io.PrintStream;
import java.lang.classfile.CodeBuilder;
import java.lang.classfile.Label;
import java.lang.classfile.TypeKind;
import java.lang.constant.ClassDesc;
import java.lang.constant.MethodTypeDesc;

/**
 * @author bin
 * @version 1.0.0
 * @since 2024/09/25
 */
@SuppressWarnings({"preview", "UnusedReturnValue"})
public interface ClassFileUtil {
    static ClassDesc of(Class<?> descriptor) {
        return ClassDesc.ofDescriptor(descriptor.descriptorString());
    }

    static MethodTypeDesc of(Class<?> returnDesc, Class<?>... paramClasses) {
        val paramDescs = new ClassDesc[paramClasses.length];
        for (int i = 0; i < paramClasses.length; i++) {
            paramDescs[i] = of(paramClasses[i]);
        }
        return MethodTypeDesc.of(of(returnDesc), paramDescs);
    }

    static CodeBuilder getstatic(CodeBuilder cb,
            Class<?> owner, String name, Class<?> type) {
        return cb.getstatic(of(owner), name, of(type));
    }

    static CodeBuilder invokevirtual(CodeBuilder cb,
            Class<?> owner, String name, MethodTypeDesc type) {
        return cb.invokevirtual(of(owner), name, type);
    }

    static CodeBuilder invokevirtual(CodeBuilder cb,
            Class<?> owner, String name, Class<?> returnDesc, Class<?>... paramClasses) {
        return cb.invokevirtual(of(owner), name, of(returnDesc, paramClasses));
    }

    static CodeBuilder invokespecial(CodeBuilder cb,
            Class<?> owner, String name, MethodTypeDesc type) {
        return cb.invokespecial(of(owner), name, type);
    }

    static CodeBuilder invokespecial(CodeBuilder cb,
            Class<?> owner, String name, Class<?> returnDesc, Class<?>... paramClasses) {
        return cb.invokespecial(of(owner), name, of(returnDesc, paramClasses));
    }

    static CodeBuilder print(CodeBuilder cb, String message) {
        getstatic(cb, System.class, "out", PrintStream.class);
        cb.ldc(message);
        return invokevirtual(cb, PrintStream.class, "print", void.class, String.class);
    }

    static CodeBuilder println(CodeBuilder cb, String message) {
        getstatic(cb, System.class, "out", PrintStream.class);
        cb.ldc(message);
        return invokevirtual(cb, PrintStream.class, "println", void.class, String.class);
    }

    static CodeBuilder callSuper(CodeBuilder cb, Class<?> owner, String name, MethodTypeDesc type) {
        cb.aload(0);
        val count = type.parameterCount();
        if (count != 0) {
            for (int i = 0; i < count; i++) {
                cb.loadLocal(TypeKind.fromDescriptor(type.parameterType(i).descriptorString()), i + 1);
            }
        }
        return invokespecial(cb,
                owner,
                name,
                type
        );
    }

    static CodeBuilder paramVariable(CodeBuilder cb, Label startScope, Label endScope, Class<?> owner,
            MethodTypeDesc type) {
        cb.localVariable(0, "this", of(owner), startScope, endScope);
        val count = type.parameterCount();
        if (count != 0) {
            for (int i = 0; i < count; i++) {
                cb.localVariable(i + 1, "var" + i, type.parameterType(i), startScope, endScope);
            }
        }
        return cb;
    }
}
