package demo.util;

import jakarta.validation.ConstraintViolation;
import jakarta.validation.Validator;
import lombok.val;
import org.springframework.validation.BindException;
import org.springframework.validation.DirectFieldBindingResult;
import org.springframework.validation.FieldError;

import java.util.Set;

/**
 * bean对象属性验证
 */
@SuppressWarnings("unused")
public class BeanValidators {
    public static void validateWithException(Validator validator, Object object) throws BindException {
        Set<ConstraintViolation<Object>> constraintViolations = validator.validate(object);
        if (constraintViolations.isEmpty()) {
            return;
        }
        val objectName = object.getClass().getSimpleName();
        val result = new DirectFieldBindingResult(object, objectName);
        for (ConstraintViolation<Object> violation : constraintViolations) {
            val fieldError = new FieldError(
                    objectName, violation.getPropertyPath().toString(),
                    violation.getMessage()
            );
            result.addError(fieldError);
        }
        throw new BindException(result);
    }
}
