package demo;

import lombok.val;

import java.io.Serializable;
import java.lang.invoke.SerializedLambda;
import java.lang.reflect.InvocationTargetException;

/**
 * @author bin
 * @since 2025/05/06
 */
public class SerializedLambdaTest {
    public static void main(String[] args)
            throws NoSuchMethodException, InvocationTargetException, IllegalAccessException {
        SFunction lambda = SerializedLambdaTest::costum;

        val lambdaClass = lambda.getClass();
        val method = lambdaClass.getDeclaredMethod("writeReplace");
        method.setAccessible(Boolean.TRUE);
        val serializedLambda = (SerializedLambda) method.invoke(lambda);
        val getterMethod = serializedLambda.getImplMethodName();

        System.out.println(getterMethod);
    }

    public interface SFunction extends Serializable {
        SerializedLambda test();
    }

    private static SerializedLambda costum() {
        return null;
    }
}
