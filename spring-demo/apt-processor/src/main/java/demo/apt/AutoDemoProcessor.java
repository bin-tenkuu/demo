package demo.apt;

import com.google.auto.service.AutoService;
import com.squareup.javapoet.JavaFile;
import com.squareup.javapoet.MethodSpec;
import com.squareup.javapoet.TypeSpec;
import org.springframework.beans.factory.InitializingBean;
import org.springframework.stereotype.Component;

import javax.annotation.processing.*;
import javax.lang.model.SourceVersion;
import javax.lang.model.element.*;
import javax.lang.model.util.Elements;
import javax.tools.Diagnostic;
import java.io.IOException;
import java.util.*;

/**
 * 注解处理器：扫描所有被 @HelloWorld 标记的类，
 * 为每个类生成一个名为 XxxHello 的可运行类，
 * 其 main 方法会打印注解中指定的消息。
 *
 * @author bin
 * @since 2026/09/21
 */
@AutoService(Processor.class)
public class AutoDemoProcessor extends AbstractProcessor {

    private Messager messager;
    private Filer filer;
    private Elements elementUtils;

    @Override
    public Set<String> getSupportedAnnotationTypes() {
        return Set.of(AutoDemo.class.getName());
    }

    @Override
    public Set<String> getSupportedOptions() {
        return Set.of();
    }

    @Override
    public SourceVersion getSupportedSourceVersion() {
        return SourceVersion.latestSupported();
    }

    @Override
    public synchronized void init(ProcessingEnvironment processingEnv) {
        super.init(processingEnv);
        this.messager = processingEnv.getMessager();
        this.filer = processingEnv.getFiler();
        this.elementUtils = processingEnv.getElementUtils();
    }

    @Override
    public boolean process(Set<? extends TypeElement> annotations, RoundEnvironment roundEnv) {
        var map = new HashMap<PackageElement, Set<TypeElement>>();
        // 遍历所有被 @AutoDemo 标记的元素
        for (Element element : roundEnv.getElementsAnnotatedWith(AutoDemo.class)) {
            process(map, null, element);
        }
        for (var entry : map.entrySet()) {
            var packageElement = entry.getKey();
            var typeElements = entry.getValue();
            var packageName = packageElement.getQualifiedName().toString();

            // 构建 main 方法
            var method = MethodSpec.methodBuilder("afterPropertiesSet")
                    .addAnnotation(Override.class)
                    .addModifiers(Modifier.PUBLIC)
                    .addException(Exception.class)
                    .returns(void.class);
            for (var typeElement : typeElements) {
                List<? extends VariableElement> parameters = null;
                for (var element : typeElement.getEnclosedElements()) {
                    if (element.getKind() != ElementKind.CONSTRUCTOR) {
                        continue;
                    }
                    var parameters1 = ((ExecutableElement) element).getParameters();
                    if (parameters == null) {
                        parameters = parameters1;
                    } else if (parameters.size() > parameters1.size()) {
                        parameters = parameters1;
                    }
                }
                if (parameters == null || parameters.isEmpty()) {
                    method.addStatement("new $T();", typeElement);
                } else {
                    var objs = new Object[1 + parameters.size()];
                    var sb = new StringBuilder();
                    sb.append("new $T(");
                    sb.repeat("($T) null, ", parameters.size());
                    sb.setLength(sb.length() - 2);
                    sb.append(");");
                    objs[0] = typeElement;
                    for (var i = 0; i < parameters.size(); i++) {
                        objs[1 + i] = parameters.get(i);
                    }
                    method.addStatement(sb.toString(), objs);

                }
            }
            // 构建类
            var buildClass = TypeSpec.classBuilder("AutoDemoBuild")
                    .addAnnotation(Component.class)
                    .addModifiers(Modifier.PUBLIC)
                    .addSuperinterface(InitializingBean.class)
                    .addMethod(method.build());

            // 构建 Java 文件并写入
            var javaFile = JavaFile.builder(packageName, buildClass.build())
                    .addFileComment("""
                            由 HelloWorldProcessor 自动生成，请勿手动修改
                            注册类数量: $L
                            """, typeElements.size());

            messager.printMessage(Diagnostic.Kind.NOTE,
                    "为包 [" + packageName + "] 生成 AutoDemoBuild 类，消息: " + packageElement);
            try {
                javaFile.build().writeTo(filer);
            } catch (IOException e) {
                messager.printMessage(Diagnostic.Kind.ERROR,
                        "生成代码失败: " + e.getMessage(), packageElement);
            }
        }
        return true; // 声明已处理该注解，其他处理器不再处理
    }

    private void process(Map<PackageElement, Set<TypeElement>> map, PackageElement packageName, Element element) {
        switch (element.getKind()) {
            case PACKAGE -> {
                var packageElement = (PackageElement) element;
                for (var enclosedElement : packageElement.getEnclosedElements()) {
                    process(map, packageElement, enclosedElement);
                }
            }
            case CLASS -> {
                TypeElement typeElement = (TypeElement) element;
                var demo = element.getAnnotation(AutoDemo.class);
                if (demo != null && demo.value()) {
                    return;
                }
                if (packageName == null) {
                    packageName = elementUtils.getPackageOf(typeElement);
                }
                map.computeIfAbsent(packageName, k -> new HashSet<>())
                        .add(typeElement);
            }
        }
    }
}
